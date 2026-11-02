# 🦜 Parakeet AI - Local Desktop Copilot (Prototype)

A fast, private, and fully local desktop copilot prototype inspired by **Parakeet AI**.

It captures live microphone audio, transcribes your speech in real time with local speech-to-text, and progressively streams intelligent suggested answers using a local LLM runtime (Ollama).

---

## ✨ Key Features

1. **Simple Desktop Window**:
   - Sleek dark aesthetic built with **CustomTkinter**.
   - Clean Start and Stop listening controls with live status indicators.
   - **Float on Top** toggle to overlay Zoom, Google Meet, or browser tabs.
   - **Stealth / Opacity Slider** for discreet usage.

2. **Real-Time Speech-to-Text**:
   - High-performance, quantized **`faster-whisper`** (`tiny.en` / `base.en`) running 100% locally on CPU (`int8`).
   - Continuous audio capture via `sounddevice` with dynamic RMS Voice Activity Detection (VAD).
   - Live **VU meter / Mic Activity visualizer** so you know when audio is being received.
   - Real-time chunk accumulation: transcription appears as you speak.
   - Editable transcription area: edit what was heard or add extra context.

3. **Progressive AI Answer Suggestion**:
   - Click **"✨ Suggest Answer"** or press <kbd>Ctrl</kbd> + <kbd>Enter</kbd>.
   - Streams answers token-by-token directly from local **Ollama** (`qwen2.5:3b`).
   - Live tokens-per-second (`tok/s`) and elapsed latency metrics.
   - **Stop Generating** button to interrupt generation at any time.
   - **One-Click Copy** buttons for both transcription and generated answer.

4. **Copilot Modes**:
   - **Direct Interview Answer**: Confident, articulate, professional first-person answer.
   - **Concise Bullet Points**: Rapid talking points, key takeaways, and critical insights.
   - **Technical Deep Dive**: Architectural trade-offs, algorithms, and code design.

5. **Quick-Test Evaluation**:
   - Built-in sample question injector so you can test answer generation immediately even without speaking.

---

## 🏗️ Architecture

```
                                      +-------------------------------+
                                      |    ParakeetApp (CustomTkinter)|
                                      +---------------+---------------+
                                                      |
                   +----------------------------------+----------------------------------+
                   |                                                                     |
                   v                                                                     v
    +------------------------------+                                      +------------------------------+
    |    AudioTranscriber          |                                      |    LocalLLMClient            |
    |  - sounddevice (16kHz mono)  |                                      |  - Ollama REST API           |
    |  - RMS Voice Activity Detect |                                      |  - Model: qwen2.5:3b         |
    |  - faster-whisper (tiny.en)  |                                      |  - Progressive Token Stream  |
    +--------------+---------------+                                      +--------------+---------------+
                   |                                                                     |
                   v                                                                     v
       [Live Transcription Panel]  ------------------------->  [Suggest Answer] ---------> [AI Answer Panel]
```

---

## 🚀 Quickstart

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **Ollama** installed and running locally:
  ```bash
  # Check if Ollama is running
  ollama list

  # Pull the default fast model (if not already pulled)
  ollama pull qwen2.5:3b
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
On Windows:
```bash
run.bat
```
Or directly with Python:
```bash
python app.py
```

---

## 🧪 Automated Tests

Run the included automated test suite to verify audio devices, Whisper transcription, and Ollama streaming:
```bash
python test_prototype.py
```

---

## 📁 Project Structure

```
.
├── app.py                  # Main desktop GUI application
├── audio_transcriber.py    # Microphone audio capture & faster-whisper STT worker
├── llm_client.py           # Ollama client with progressive token streaming & presets
├── test_audio_generator.py # Sample questions and synthetic audio tone synthesis
├── test_prototype.py       # Automated verification test suite
├── requirements.txt        # Python dependencies
├── run.bat                 # 1-click Windows launcher
└── README.md               # Documentation
```
