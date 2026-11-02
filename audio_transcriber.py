"""
Audio Transcriber Module for Parakeet AI Prototype
Captures microphone audio locally using sounddevice and transcribes in real-time
using faster-whisper (CTranslate2).
"""

import threading
import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Callable, Optional, List, Dict


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "tiny.en",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 2.0,
        energy_threshold: float = 0.015,
        on_transcription_callback: Optional[Callable[[str, bool], None]] = None,
        on_level_callback: Optional[Callable[[float], None]] = None,
        on_status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model_size = model_size
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.energy_threshold = energy_threshold

        self.on_transcription = on_transcription_callback
        self.on_level = on_level_callback
        self.on_status = on_status_callback

        self.model: Optional[WhisperModel] = None
        self.is_recording = False
        self._audio_queue = queue.Queue()
        self._record_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._init_model()

    def _init_model(self):
        """Loads Whisper model with CPU int8 quantization."""
        if self.on_status:
            self.on_status(f"Loading Whisper model ({self.model_size})...")
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            if self.on_status:
                self.on_status(f"Whisper model ready ({self.model_size})")
        except Exception as e:
            if self.on_status:
                self.on_status(f"Error loading model: {e}")
            raise e

    def switch_model(self, model_size: str):
        """Switch between tiny.en and base.en."""
        with self._model_lock:
            self.model_size = model_size
            self._init_model()

    @staticmethod
    def get_input_devices() -> List[Dict[str, any]]:
        """Returns list of available microphone input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            default_input = sd.default.device[0]
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Device {idx}")
                    devices.append({
                        "id": idx,
                        "name": name,
                        "hostapi": dev.get("hostapi"),
                        "is_default": (idx == default_input),
                    })
        except Exception as e:
            print(f"Error querying audio devices: {e}")
        return devices

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice InputStream."""
        if not self.is_recording:
            return
        
        # Audio data is float32 [-1.0, 1.0]
        audio_chunk = indata[:, 0].copy()
        
        # Calculate RMS energy for visualizer
        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        if self.on_level:
            try:
                self.on_level(rms)
            except Exception:
                pass

        self._audio_queue.put(audio_chunk)

    def start(self, device_index: Optional[int] = None):
        """Starts capturing and transcribing microphone audio."""
        with self._lock:
            if self.is_recording:
                return

            if device_index is not None:
                self.device_index = device_index

            self.is_recording = True
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                # Open audio stream
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.device_index,
                    blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception as e:
                self.is_recording = False
                if self.on_status:
                    self.on_status(f"Mic error: {e}")
                raise e

            # Start worker thread
            self._worker_thread = threading.Thread(
                target=self._transcribe_worker, daemon=True
            )
            self._worker_thread.start()

            if self.on_status:
                self.on_status("Listening to microphone...")

    def stop(self):
        """Stops capturing audio and transcribes any pending audio buffer."""
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False

            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            if self.on_level:
                self.on_level(0.0)

            if self.on_status:
                self.on_status("Stopped listening")

    def _transcribe_worker(self):
        """Processes audio chunks from queue and runs transcription."""
        buffer = []
        silence_frames = 0
        speech_detected = False
        min_speech_frames = int(0.4 * 10)  # at least 400ms of audio
        max_buffer_frames = int(4.0 * 10)  # max 4 seconds per segment

        while self.is_recording or not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                if not self.is_recording:
                    break
                continue

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            buffer.append(chunk)

            if rms > self.energy_threshold:
                speech_detected = True
                silence_frames = 0
            else:
                if speech_detected:
                    silence_frames += 1

            # Trigger transcription if:
            # 1. We had speech and now experienced ~0.8s of silence (natural pause)
            # 2. Or buffer exceeded max_buffer_frames (4 seconds)
            # 3. Or user clicked Stop and buffer has remaining speech
            should_transcribe = False
            if speech_detected and len(buffer) >= min_speech_frames:
                if silence_frames >= 8:  # ~0.8s of silence
                    should_transcribe = True
                elif len(buffer) >= max_buffer_frames:
                    should_transcribe = True

            if not self.is_recording and speech_detected and len(buffer) > min_speech_frames:
                should_transcribe = True

            if should_transcribe and len(buffer) > 0:
                audio_np = np.concatenate(buffer)
                buffer = []
                silence_frames = 0
                speech_detected = False
                self._run_transcription(audio_np)

        # Final flush if anything remains
        if speech_detected and len(buffer) > min_speech_frames:
            audio_np = np.concatenate(buffer)
            self._run_transcription(audio_np)

    def _run_transcription(self, audio_data: np.ndarray):
        """Executes Whisper model transcription on the audio segment."""
        if self.model is None or len(audio_data) < int(self.sample_rate * 0.3):
            return

        try:
            with self._model_lock:
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=400),
                )
                text = " ".join([seg.text.strip() for seg in segments]).strip()

            if text and self.on_transcription:
                self.on_transcription(text, True)
        except Exception as e:
            print(f"Transcription error: {e}")

    def transcribe_audio_file_or_array(self, audio_data: np.ndarray) -> str:
        """Utility method to directly transcribe an audio array."""
        if self.model is None:
            return ""
        with self._model_lock:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=1,
                language="en",
            )
            return " ".join([seg.text.strip() for seg in segments]).strip()
