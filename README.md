# Real-Time-Transaural-Audio-Reconstruction-Spatialization-for-Embedded-Edge-AI
Utilizing HRTF and ITD/ILD cues for spatial realism.mplementing Recursive XTC to solve the "crosstalk" problem.Optimized Depthwise Separable Convolutions for real-time inference on lower-power processors.Managing the path from DAC to Power Amplifiers for stable loudspeaker driving.
# Real-Time Transaural Audio Reconstruction & Spatialization

## 3D Soundscape Design for Mudumalai National Park
An end-to-end system designed to recreate the immersive experience of Mudumalai National Park using only two loudspeakers. This project integrates **Edge-AI source separation** with **Transaural DSP** to provide a dynamic 3D soundscape that responds to listener movement.

---

## 🛠 System Architecture
The system is divided into three high-level modules designed for low-latency embedded execution:

1.  **Sensing & State Estimation:** * Camera-based tracking (BlazeFace/MobileNet) for head position and yaw.
    * Kalman filtering for jitter suppression and temporal stability.
2.  **DSP Engine (Transaural Pipeline):**
    * **Binaural Synthesis:** Applies HRTF, ITD, and ILD cues to mono sources.
    * **Crosstalk Cancellation (XTC):** Recursive filtering to ensure binaural cues are delivered correctly over loudspeakers.
3.  **Edge-AI Separation:**
    * **Lightweight Model:** Uses Depthwise Separable Convolutions to reduce parameter count for ARM Cortex-M/ESP32 compatibility.
    * **Phase-Preservation:** Utilizes Magnitude-Masking with phase-sensitive mapping to prevent metallic artifacts.


## 🚀 Key Features
* **Time-Domain Integrity:** Focused on preserving transient clarity and phase alignment, meeting the "Time" precision standards of Atomik Audio.
* **Recursive XTC:** Solves the acoustic "leakage" problem in 2.0 speaker setups, creating a stable virtual world rather than fixed physical sound sources.
* **Resource Optimized:** Redesigned from a heavy U-Net to a mobile-first architecture, optimized for real-time inference on edge devices.
* **Distance Rendering:** Simulates depth through frequency-dependent gain reduction and low-pass filtering for distant sources.

## 📂 Project Structure
* `/models`: Contains `TinySeparator`, a DSC-based lightweight neural network.
* `/dsp`: Implementation of Recursive XTC and HRTF filtering.
* `/sensing`: Listener state estimation logic using camera/IMU fusion.

---

## 🔧 Installation & Usage
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Inference:**
    ```bash
    python main.py --mode test --audio_path ./input/nature_mix.wav
    ```

## 🎯 Target Hardware
Designed for deployment on high-performance embedded platforms such as **Jetson Nano**, **Raspberry Pi 4**, or **ESP32-S3** (with quantization).
