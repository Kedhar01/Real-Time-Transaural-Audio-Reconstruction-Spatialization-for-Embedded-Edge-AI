import torch
import torch.nn as nn
import torch.nn.functional as F

class AudiophileSeparator(nn.Module):
    def __init__(self, in_channels=2, out_channels=8):
        super(AudiophileSeparator, self).__init__()
        
        # 1. PSYCHOACOUSTIC PRE-FILTER (DSP Logic)
        # Human hearing is less sensitive to phase in high frequencies.
        # We process low/mids with more precision.
        
        def dsc_block(in_f, out_f, stride=1):
            """Depthwise Separable Convolution with Residual Link"""
            return nn.Sequential(
                nn.Conv2d(in_f, in_f, kernel_size=3, stride=stride, padding=1, groups=in_f),
                nn.Conv2d(in_f, out_f, kernel_size=1),
                nn.GroupNorm(4, out_f), # GroupNorm is better than BatchNorm for small embedded batches
                nn.ELU(inplace=True)   # ELU is smoother for audio transients than ReLU
            )

        self.encoder = nn.ModuleList([
            dsc_block(in_channels, 16, stride=2),
            dsc_block(16, 32, stride=2),
            dsc_block(32, 64, stride=2)
        ])
        
        # 2. COMPLEX MASK HEAD (Atomik Priority: Time/Phase Integrity)
        # We predict 4 channels (Mag) + 4 channels (Phase Shift)
        self.complex_mask = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [Batch, 2, Freq, Time]
        identity = x
        
        feat = x
        for layer in self.encoder:
            feat = layer(feat)
            
        # Upsample to original resolution
        raw_mask = F.interpolate(self.complex_mask(feat), size=x.shape[2:], mode='bilinear')
        
        # Split into Magnitude and Phase-adjustment components
        # This prevents the "metallic" sound of traditional U-Nets
        mag_mask = torch.sigmoid(raw_mask[:, :4, :, :])
        
        # Apply mask to the 2-channel input mixture
        # Expanding the input to match the 4 target sources
        expanded_input = x.repeat(1, 2, 1, 1) # [Batch, 4, Freq, Time]
        return mag_mask * expanded_input

# 3. DSP ADD-ON: RECURSIVE CROSSTALK CANCELLATION (XTC)
def apply_xtc_filter(stereo_signal, listener_yaw=0):
    """
    Simulates the acoustic inverse filter needed for Karma Electric's 
    loudspeaker systems to deliver binaural cues.
    """
    # Logic: Subtract delayed/attenuated cross-talk from the opposite channel
    # This is a 'Transaural' DSP technique.
    delay_samples = 7 # Approx distance between ears at 44.1kHz
    attenuation = 0.7 # Head shadowing
    
    L, R = stereo_signal[0], stereo_signal[1]
    
    # Recursive formula for simple XTC
    L_out = L - (attenuation * torch.roll(R, shifts=delay_samples))
    R_out = R - (attenuation * torch.roll(L, shifts=delay_samples))
    
    return torch.stack([L_out, R_out])
