# Supernan AI Intern – Video Dubbing Pipeline

Python pipeline to dub a video segment into Hindi: **ingest → transcribe → translate → voice clone → lip-sync**. Built for the Supernan AI Automation Intern challenge, using free/open-source tools and designed to run on **Google Colab (free tier)**, Kaggle, or local GPU.

## What it does

1. **Ingest** – Downloads or reads the source video and extracts a segment (default **0:15–0:30** for the “Golden 15 Seconds”).
2. **Transcribe** – Extracts audio and transcribes with **Whisper** (open-source).
3. **Translate** – Translates English → Hindi (default: **Helsinki-NLP/opus-mt-en-hi**; optional IndicTrans2 for higher quality).
4. **Generate Hindi audio** – **Coqui XTTS v2** voice cloning from the original speaker.
5. **Lip-sync** – **Wav2Lip** to sync the video face to the new Hindi audio.

Output: a Hindi-dubbed video clip with cloned voice and lip-sync.

---

## Setup

### 1. Clone and Python environment

- **Python**: 3.10+ recommended.
- Create a venv and install dependencies:

```bash
cd supernan
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **ffmpeg** must be installed (for segment extraction and final muxing):
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`

### 2. Source video

- **Option A**: Use the default Supernan training video. The script will download it from Google Drive (link in the challenge).
- **Option B**: Use a local file:
  ```bash
  python dub_video.py --input_video /path/to/your/video.mp4
  ```

### 3. Wav2Lip (lip-sync)

Lip-sync uses the [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) repo. You need to clone it and install its dependencies, then point the pipeline to it:

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
pip install -r requirements.txt
# Download checkpoints (see Wav2Lip README), e.g. wav2lip_gan.pth into checkpoints/
```

Then either:

- Set env var: `export WAV2LIP_ROOT=/path/to/Wav2Lip`
- Or pass: `python dub_video.py --wav2lip_root /path/to/Wav2Lip`

If `WAV2LIP_ROOT` is not set and `--wav2lip_root` is not given, the pipeline will run up to TTS and then **skip lip-sync**, producing a video with Hindi audio only (no lip-sync). Useful for quick tests.

### 4. Google Colab (free tier)

- Upload this repo (or clone from GitHub) into Colab.
- Install dependencies in a cell:
  ```bash
  !pip install -r requirements.txt
  ```
- Install ffmpeg: `!apt install -y ffmpeg`
- Clone Wav2Lip in the notebook and download its checkpoint(s).
- Set `WAV2LIP_ROOT` to the Wav2Lip path in the notebook.
- Run:
  ```bash
  !python dub_video.py --work_dir /content/workdir --start 15 --end 30
  ```
- Use a **GPU runtime** for Whisper, XTTS, and Wav2Lip (faster).

---

## Usage

**Default (0:15–0:30 segment, auto-download source):**

```bash
python dub_video.py
```

**Custom segment and output:**

```bash
python dub_video.py --input_video video.mp4 --start 10 --end 25 --output my_dub.mp4
```

**Resume / skip steps** (e.g. already have transcript and translation):

```bash
python dub_video.py --skip_steps ingest,transcribe,translate
```

**Better Whisper quality** (slower):

```bash
python dub_video.py --whisper_model medium
```

**All options:**

```text
--input_video    Path to source video (omit to use Drive download)
--work_dir       Working dir for intermediates (default: workdir)
--start          Segment start in seconds (default: 15)
--end            Segment end in seconds (default: 30)
--output         Output video path
--skip_steps     Comma-separated: ingest,transcribe,translate,tts,lipsync
--wav2lip_root   Path to Wav2Lip repo
--drive_id       Google Drive file ID (if not default)
--whisper_model  tiny|base|small|medium|large
--wav2lip_checkpoint  e.g. wav2lip_gan
```

---

## Dependencies (summary)

| Step        | Tool / model                          | Purpose           |
|------------|----------------------------------------|-------------------|
| Ingest     | ffmpeg, gdown                          | Segment + download |
| Transcribe | OpenAI Whisper                         | EN transcription   |
| Translate  | Helsinki-NLP/opus-mt-en-hi (or IndicTrans2) | EN → HI        |
| TTS        | Coqui TTS (XTTS v2)                    | Hindi voice clone  |
| Lip-sync   | Wav2Lip (separate repo)                | Lip-sync video     |

See `requirements.txt` for pip packages. Wav2Lip is not on PyPI; clone its repo and set `WAV2LIP_ROOT`.

---

## Estimated cost per minute of video (if scaled)

Assumptions: running on **free Colab** or self-hosted GPU; no paid APIs.

- **Compute**: ₹0 if using Colab free tier / Kaggle / own GPU.
- **APIs**: ₹0 (Whisper, Helsinki-NLP, XTTS, Wav2Lip are open-source).
- **Estimated cost per minute of output**: **₹0** at current design.

If you later switch to paid services (e.g. cloud GPU, ElevenLabs, paid translation API), you’d add:
- Cloud GPU: order of ~$0.5–2 per hour → scale by (minutes of video × pipeline runtime per minute).
- Paid TTS/translation: per character/minute pricing from the provider.

---

## Known limitations

- **Lip-sync**: Wav2Lip can sometimes blur or soften the face; for higher visual fidelity, **VideoReTalking** or **GFPGAN/CodeFormer** post-processing can be used (not included in this repo).
- **Translation**: Default model (Helsinki-NLP) is good but not always “nanny-level” natural; for best quality, integrate **IndicTrans2** (En→Indic) and optionally tune for childcare domain.
- **Long videos**: Pipeline is written for a single segment; full-video scaling would need batching (e.g. by sentence or fixed duration), silence handling, and possibly async/queue for TTS and lip-sync.
- **Audio length**: XTTS output duration may not exactly match the original segment length; the script does not yet force duration alignment (e.g. time-stretch) before lip-sync; you may see minor sync drift on long segments.
- **Wav2Lip**: Requires a visible, front-facing face in the segment; profile or heavily occluded faces may fail or look worse.

---

## Possible improvements (with more time)

- **Visual fidelity**: Add **VideoReTalking** or **Wav2Lip + GFPGAN/CodeFormer** face restoration.
- **Translation**: Replace or complement with **IndicTrans2** (and optionally fine-tune for childcare/supernan context).
- **Sync**: Stretch or trim generated Hindi audio to match original segment duration before lip-sync.
- **Full-video**: Add batch loop over segments (e.g. by Whisper segments), with silence detection and optional parallelisation for TTS/lip-sync.
- **Scale (500 hours overnight)**: Design for distributed jobs (e.g. Celery + GPU workers, or cloud batch), chunking by segment, and cost vs. deadline trade-offs (e.g. more GPUs vs. cheaper/slower runs).

---

## Submission (challenge)

- **Output**: A 15–30 second dubbed clip (e.g. `workdir/dubbed_output.mp4`).
- **Repo**: This codebase, with clear commits and this README.
- **Loom**: Explain pipeline, resourceful choices (free Colab, open-source stack), and how you’d scale to 500 hours with a budget.

---

## License

Use and modify as needed for the Supernan AI Intern challenge and your portfolio.
