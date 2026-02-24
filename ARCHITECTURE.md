# Technical Architecture: Supernan Hindi Dubbing Pipeline 🎙️

This document outlines the architectural design, technical optimizations, and scalability patterns implemented in the Supernan automated video dubbing system.

---

## 🏗️ 1. Core Architecture
The pipeline follows a modular **"Capture-to-Sync"** flow, designed for high reuse and error recovery:

1.  **Ingest**: Downloads source video (from Google Drive/Local) and extracts the targeted segment using `ffmpeg` (e.g., the 15-30s "Golden Segment").
2.  **Transcribe**: Uses **OpenAI Whisper** for high-accuracy ASR (Speech-to-Text).
3.  **Translate**: English-to-Hindi translation via **Helsinki-NLP opus-mt**.
4.  **Voice Cloning (TTS)**: **Coqui XTTS v2** clones the original speaker's voice to generate the Hindi audio track.
5.  **Lip-Sync**: **Wav2Lip** synchronizes the video face movement to the new Hindi audio.

---

## 🚀 2. Engineering Highlights & Optimizations
These features ensure production-grade robustness and resource efficiency:

### 🧩 Robustness & Error Handling
- **Wav2Lip Face Persistence Patch**: I modified `inference.py` to fix a common Wav2Lip failure. If the model loses track of a face in a frame (due to occlusion or turning), it now "persists" the last known bounding box instead of crashing.
- **Silent Segment Handling**: Added logic to detect and handle silent or non-speech segments at the start/end of clips, preventing ASR/TTS sync errors.
- **Automatic Language Detection**: Implemented a multi-window voting mechanism to identify source languages (e.g., English vs. Kannada) automatically.

### ⚡ Resource Optimization (OOM Prevention)
- **Dynamic Resizing**: Added a `--resize_factor` parameter to downscale frames before processing, drastically reducing VRAM usage on free Colab T4 GPUs.
- **Batch Processing**: Configurable `wav2lip_batch_size` allows the pipeline to run even on low-memory environments without triggering Out-of-Memory (OOM) errors.

---

## 💰 3. Economics & Sustainability
- **Current Cost**: **₹0 per minute**.
- **Stack**: 100% Open-Source (OpenAI Whisper, Coqui, Wav2Lip, Helsinki-NLP).
- **Hosting**: Designed for **Google Colab Free Tier**, Kaggle, or local entry-level GPUs.

---

## 📈 4. Scaling to 500 Hours
To process 500 hours of video overnight, the architecture would shift to:

1.  **Distributed Workers**: Move from a single script to a **Celery/RabbitMQ** queue system with multiple GPU nodes.
2.  **Parallel Chunking**: Video processing is "embarrassingly parallel." We divide full videos into 1-minute chunks, process them concurrently, and stitch them back with `ffmpeg` concat.
3.  **Infrastructure**: Utilize **Spot Instances** (AWS/GCP) to minimize costs, with a fallback to higher-quality paid APIs (e.g., ElevenLabs) only for "High-Priority" speakers.

---

## ✨ 5. Future Enhancements
- **Face Restoration**: Integrate **GFPGAN** or **CodeFormer** to upscale the blurred lip-sync region.
- **Tonal Translation**: Fine-tune the translation model specifically for "Childcare/Caregiver" vocabulary to match the Supernan brand voice.
