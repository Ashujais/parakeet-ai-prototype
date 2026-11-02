"""
Comprehensive automated test suite for Parakeet AI Prototype.
Tests:
1. Audio devices & capture
2. Faster-Whisper local transcription
3. Ollama local LLM streaming & token callbacks
4. UI initialization sanity
"""

import sys
import time
import numpy as np
from audio_transcriber import AudioTranscriber
from llm_client import LocalLLMClient
from test_audio_generator import SAMPLE_QUESTIONS, generate_synthetic_tone


def test_audio_device_discovery():
    print("\n[TEST 1] Testing audio device discovery...")
    devices = AudioTranscriber.get_input_devices()
    print(f"Found {len(devices)} input audio device(s).")
    for d in devices[:3]:
        print(f" - [{d['id']}] {d['name']} (default={d['is_default']})")
    assert len(devices) > 0, "No audio input devices found!"
    print("[PASS] Audio device discovery passed.")


def test_whisper_transcription():
    print("\n[TEST 2] Testing Faster-Whisper local transcription...")
    transcriber = AudioTranscriber(model_size="tiny.en")
    tone = generate_synthetic_tone(duration_sec=1.5)
    # Transcribing synthetic tone (silence or tone should complete without crashing)
    result = transcriber.transcribe_audio_file_or_array(tone)
    print(f"Whisper inference completed! Result text: '{result}'")
    print("[PASS] Faster-Whisper test passed.")


def test_ollama_streaming():
    print("\n[TEST 3] Testing Ollama streaming answer generation...")
    client = LocalLLMClient()
    health = client.check_health()
    print(f"Ollama health: {health}")
    assert health.get("connected"), f"Ollama is not connected: {health.get('error')}"

    tokens_received = []
    done_event = False

    def on_token(t):
        tokens_received.append(t)
        sys.stdout.write(t)
        sys.stdout.flush()

    def on_done(full_text):
        nonlocal done_event
        done_event = True

    print("Sending prompt: 'What is a binary search tree in 15 words?'")
    client.stream_answer(
        question="What is a binary search tree in 15 words?",
        mode="Concise Bullet Points",
        on_token=on_token,
        on_done=on_done,
    )

    # Wait up to 30 seconds for completion
    start = time.time()
    while not done_event and time.time() - start < 30:
        time.sleep(0.2)

    print()
    assert len(tokens_received) > 0, "No tokens received from Ollama streaming!"
    print(f"[PASS] Ollama streaming passed! Received {len(tokens_received)} tokens.")


def test_ui_instantiation():
    print("\n[TEST 4] Testing UI instantiation...")
    from app import ParakeetApp
    app = ParakeetApp()
    app.update()  # Processes window creation and layout
    print("[PASS] ParakeetApp GUI instantiated and layout evaluated successfully!")
    app.destroy()


if __name__ == "__main__":
    print("========================================")
    print("Running Parakeet AI Prototype Test Suite")
    print("========================================")
    test_audio_device_discovery()
    test_whisper_transcription()
    test_ollama_streaming()
    test_ui_instantiation()
    print("\n========================================")
    print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
    print("========================================")
