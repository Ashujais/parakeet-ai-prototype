"""
Parakeet AI Prototype - Desktop Application
A local, real-time AI copilot for microphone audio transcription and progressive answer generation.

Features:
- Live microphone audio capture with sounddevice
- Real-time local speech-to-text with faster-whisper (tiny.en / base.en)
- Local LLM inference with Ollama (qwen2.5:3b)
- Progressive streaming token-by-token answer generation
- Sleek dark UI with CustomTkinter, VU meter, device selection, and stealth opacity
"""

import sys
import time
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

from audio_transcriber import AudioTranscriber
from llm_client import LocalLLMClient
from test_audio_generator import SAMPLE_QUESTIONS


# CustomTkinter Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class ParakeetApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Parakeet AI - Desktop Copilot (Prototype)")
        self.geometry("980x780")
        self.minsize(860, 680)

        # Application State
        self.transcriber: AudioTranscriber = None
        self.llm_client = LocalLLMClient()
        self.is_listening = False
        self.is_generating = False
        self.start_generation_time = 0.0
        self.token_count = 0
        self.selected_device_id = None
        self.full_answer_text = ""

        # Build UI Components
        self._create_header()
        self._create_control_panel()
        self._create_main_content()
        self._create_footer()

        # Keyboard shortcuts
        self.bind("<Control-Return>", lambda e: self.on_suggest_answer())

        # Initialize background services
        self.after(200, self._async_init_services)

    # -------------------------------------------------------------------------
    # UI Building Methods
    # -------------------------------------------------------------------------

    def _create_header(self):
        """Header bar with branding, status badges, and stealth/window toggles."""
        header_frame = ctk.CTkFrame(self, fg_color="#111827", corner_radius=0, height=64)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        # Left branding
        brand_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        brand_box.pack(side="left", padx=20, pady=10)

        title_label = ctk.CTkLabel(
            brand_box,
            text="🦜 PARAKEET AI",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#10b981",
        )
        title_label.pack(side="left", padx=(0, 10))

        badge = ctk.CTkLabel(
            brand_box,
            text="LOCAL PROTOTYPE",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#1f2937",
            text_color="#9ca3af",
            corner_radius=6,
            padx=8,
            pady=2,
        )
        badge.pack(side="left")

        # Right utilities (Always on Top, Opacity, Status Badges)
        util_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        util_box.pack(side="right", padx=20, pady=10)

        # STT Badge
        self.stt_badge = ctk.CTkLabel(
            util_box,
            text="STT: Loading...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#1f2937",
            text_color="#9ca3af",
            corner_radius=6,
            padx=10,
            pady=4,
        )
        self.stt_badge.pack(side="left", padx=5)

        # LLM Badge
        self.llm_badge = ctk.CTkLabel(
            util_box,
            text="LLM: Checking...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#1f2937",
            text_color="#9ca3af",
            corner_radius=6,
            padx=10,
            pady=4,
        )
        self.llm_badge.pack(side="left", padx=5)

        # Always-on-top switch
        self.pin_var = ctk.BooleanVar(value=False)
        self.pin_switch = ctk.CTkSwitch(
            util_box,
            text="Float on Top",
            font=ctk.CTkFont(size=12),
            variable=self.pin_var,
            command=self._toggle_always_on_top,
            progress_color="#10b981",
        )
        self.pin_switch.pack(side="left", padx=10)

        # Window Opacity Slider
        opacity_label = ctk.CTkLabel(util_box, text="Opacity:", font=ctk.CTkFont(size=11), text_color="#9ca3af")
        opacity_label.pack(side="left", padx=(5, 2))

        self.opacity_slider = ctk.CTkSlider(
            util_box,
            from_=0.5,
            to=1.0,
            number_of_steps=10,
            width=90,
            command=self._set_opacity,
            progress_color="#10b981",
        )
        self.opacity_slider.set(1.0)
        self.opacity_slider.pack(side="left", padx=2)

    def _create_control_panel(self):
        """Top action panel containing Start/Stop, VU meter, mic selection & test tools."""
        panel = ctk.CTkFrame(self, fg_color="#1f2937", corner_radius=10)
        panel.pack(fill="x", padx=16, pady=(12, 8))

        # Main button row
        row = ctk.CTkFrame(panel, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=12)

        # Start button
        self.start_btn = ctk.CTkButton(
            row,
            text="🎙  Start Listening",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            width=160,
            height=40,
            corner_radius=8,
            command=self.on_start_listening,
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        # Stop button
        self.stop_btn = ctk.CTkButton(
            row,
            text="⏹  Stop Listening",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            width=160,
            height=40,
            corner_radius=8,
            state="disabled",
            command=self.on_stop_listening,
        )
        self.stop_btn.pack(side="left", padx=(0, 15))

        # Live VU Meter / Level Visualizer
        vu_box = ctk.CTkFrame(row, fg_color="transparent")
        vu_box.pack(side="left", fill="x", expand=True, padx=10)

        vu_label = ctk.CTkLabel(
            vu_box,
            text="Mic Activity Level:",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
            anchor="w",
        )
        vu_label.pack(anchor="w")

        self.vu_bar = ctk.CTkProgressBar(
            vu_box,
            orientation="horizontal",
            height=12,
            progress_color="#10b981",
            fg_color="#374151",
        )
        self.vu_bar.set(0.0)
        self.vu_bar.pack(fill="x", pady=(2, 0))

        # Audio Device Selector Dropdown
        self.device_combo = ctk.CTkComboBox(
            row,
            values=["Default Microphone"],
            width=200,
            height=36,
            corner_radius=8,
            command=self._on_device_changed,
        )
        self.device_combo.pack(side="left", padx=(10, 10))

        # Clear button
        self.clear_btn = ctk.CTkButton(
            row,
            text="Clear",
            width=70,
            height=36,
            fg_color="#374151",
            hover_color="#4b5563",
            corner_radius=8,
            command=self.on_clear_all,
        )
        self.clear_btn.pack(side="right", padx=(5, 0))

        # Quick Test Dropdown for immediate evaluation without speaking
        self.sample_combo = ctk.CTkComboBox(
            row,
            values=[q["title"] for q in SAMPLE_QUESTIONS],
            width=170,
            height=36,
            corner_radius=8,
            command=self._on_sample_question_selected,
        )
        self.sample_combo.set("Sample Question")
        self.sample_combo.pack(side="right", padx=(5, 5))

    def _create_main_content(self):
        """Split view with Live Transcription on left/top and AI Suggestion on right/bottom."""
        self.content_paned = ctk.CTkFrame(self, fg_color="transparent")
        self.content_paned.pack(fill="both", expand=True, padx=16, pady=4)

        # Configure 2-column responsive grid
        self.content_paned.grid_columnconfigure(0, weight=1, uniform="group1")
        self.content_paned.grid_columnconfigure(1, weight=1, uniform="group1")
        self.content_paned.grid_rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # LEFT PANEL: Live Transcription
        # ---------------------------------------------------------------------
        left_card = ctk.CTkFrame(self.content_paned, fg_color="#161b22", corner_radius=10, border_width=1, border_color="#30363d")
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        # Header for Left Card
        left_header = ctk.CTkFrame(left_card, fg_color="transparent")
        left_header.pack(fill="x", padx=14, pady=(12, 6))

        left_title = ctk.CTkLabel(
            left_header,
            text="🎙  Transcribed Speech / Question",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f3f4f6",
        )
        left_title.pack(side="left")

        self.mic_status_dot = ctk.CTkLabel(
            left_header,
            text="● IDLE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#9ca3af",
        )
        self.mic_status_dot.pack(side="right")

        # Editable Transcription Text Area
        self.transcription_text = ctk.CTkTextbox(
            left_card,
            wrap="word",
            font=ctk.CTkFont(family="Consolas" if sys.platform == "win32" else "Courier", size=13),
            fg_color="#0d1117",
            text_color="#e5e7eb",
            border_width=1,
            border_color="#21262d",
            corner_radius=8,
        )
        self.transcription_text.pack(fill="both", expand=True, padx=14, pady=6)
        self.transcription_text.insert(
            "1.0",
            "Click 'Start Listening' to capture your microphone audio.\n"
            "As you speak, your speech is transcribed here locally in real time.\n\n"
            "Tip: You can also edit or paste text here directly and click 'Suggest Answer'."
        )

        # Action bar at bottom of left panel
        left_action_bar = ctk.CTkFrame(left_card, fg_color="transparent")
        left_action_bar.pack(fill="x", padx=14, pady=(6, 12))

        # Mode preset dropdown
        self.mode_combo = ctk.CTkComboBox(
            left_action_bar,
            values=["Direct Interview Answer", "Concise Bullet Points", "Technical Deep Dive"],
            width=180,
            height=38,
            corner_radius=8,
        )
        self.mode_combo.set("Direct Interview Answer")
        self.mode_combo.pack(side="left", padx=(0, 8))

        # Suggest Answer Button
        self.suggest_btn = ctk.CTkButton(
            left_action_bar,
            text="✨  Suggest Answer (Ctrl+Enter)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=38,
            corner_radius=8,
            command=self.on_suggest_answer,
        )
        self.suggest_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # Copy transcription button
        self.copy_transcript_btn = ctk.CTkButton(
            left_action_bar,
            text="📋 Copy",
            width=70,
            height=38,
            fg_color="#374151",
            hover_color="#4b5563",
            corner_radius=8,
            command=lambda: self._copy_to_clipboard(self.transcription_text.get("1.0", "end-1c"), "Transcription copied!"),
        )
        self.copy_transcript_btn.pack(side="right")

        # ---------------------------------------------------------------------
        # RIGHT PANEL: Progressive AI Suggested Answer
        # ---------------------------------------------------------------------
        right_card = ctk.CTkFrame(self.content_paned, fg_color="#161b22", corner_radius=10, border_width=1, border_color="#30363d")
        right_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)

        # Header for Right Card
        right_header = ctk.CTkFrame(right_card, fg_color="transparent")
        right_header.pack(fill="x", padx=14, pady=(12, 6))

        right_title = ctk.CTkLabel(
            right_header,
            text="💡  AI Suggested Answer",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f3f4f6",
        )
        right_title.pack(side="left")

        self.stream_status_label = ctk.CTkLabel(
            right_header,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
        )
        self.stream_status_label.pack(side="right")

        # Progressive Answer Text Box
        self.answer_text = ctk.CTkTextbox(
            right_card,
            wrap="word",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color="#0d1117",
            text_color="#f9fafb",
            border_width=1,
            border_color="#21262d",
            corner_radius=8,
        )
        self.answer_text.pack(fill="both", expand=True, padx=14, pady=6)
        self.answer_text.insert(
            "1.0",
            "AI answers will stream here progressively as tokens are generated.\n\n"
            "1. Speak or select a sample question on the left.\n"
            "2. Click 'Suggest Answer' or press Ctrl+Enter.\n"
            "3. Watch the local Ollama LLM construct the answer in real time!"
        )

        # Action bar at bottom of right panel
        right_action_bar = ctk.CTkFrame(right_card, fg_color="transparent")
        right_action_bar.pack(fill="x", padx=14, pady=(6, 12))

        # Stop Generation button
        self.stop_gen_btn = ctk.CTkButton(
            right_action_bar,
            text="⏹  Stop Generating",
            font=ctk.CTkFont(size=12),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=38,
            corner_radius=8,
            state="disabled",
            command=self.on_stop_generation,
        )
        self.stop_gen_btn.pack(side="left", padx=(0, 10))

        # Stats label (tokens, words, tok/s)
        self.stats_label = ctk.CTkLabel(
            right_action_bar,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
        )
        self.stats_label.pack(side="left", padx=5)

        # Copy Answer Button
        self.copy_answer_btn = ctk.CTkButton(
            right_action_bar,
            text="📋 Copy Answer",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            width=120,
            height=38,
            corner_radius=8,
            command=lambda: self._copy_to_clipboard(self.answer_text.get("1.0", "end-1c"), "Answer copied to clipboard!"),
        )
        self.copy_answer_btn.pack(side="right")

    def _create_footer(self):
        """Status footer at the bottom."""
        footer = ctk.CTkFrame(self, fg_color="#111827", corner_radius=0, height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self.footer_status = ctk.CTkLabel(
            footer,
            text="System initializing...",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
        )
        self.footer_status.pack(side="left", padx=16)

        privacy_label = ctk.CTkLabel(
            footer,
            text="🔒 100% Offline & Local  •  CTranslate2 Whisper + Ollama",
            font=ctk.CTkFont(size=11),
            text_color="#6b7280",
        )
        privacy_label.pack(side="right", padx=16)

    # -------------------------------------------------------------------------
    # Asynchronous Initialization
    # -------------------------------------------------------------------------

    def _async_init_services(self):
        """Initializes Whisper STT and verifies Ollama connectivity in background."""
        def init_task():
            # 1. Check Ollama
            ollama_health = self.llm_client.check_health()
            if ollama_health.get("connected"):
                model_name = ollama_health.get("current_model", "qwen2.5:3b")
                self.after(0, lambda: self.llm_badge.configure(
                    text=f"LLM: {model_name} (Ready)",
                    fg_color="#064e3b",
                    text_color="#34d399",
                ))
            else:
                self.after(0, lambda: self.llm_badge.configure(
                    text="LLM: Ollama Offline",
                    fg_color="#7f1d1d",
                    text_color="#f87171",
                ))

            # 2. Query audio devices
            devices = AudioTranscriber.get_input_devices()
            device_names = [f"[{d['id']}] {d['name'][:30]}" for d in devices]
            default_id = next((d["id"] for d in devices if d["is_default"]), (devices[0]["id"] if devices else None))

            def update_devices():
                if device_names:
                    self.device_combo.configure(values=device_names)
                    default_name = next((f"[{d['id']}] {d['name'][:30]}" for d in devices if d["is_default"]), device_names[0])
                    self.device_combo.set(default_name)
                    self.selected_device_id = default_id

            self.after(0, update_devices)

            # 3. Load faster-whisper model
            try:
                self.after(0, lambda: self.footer_status.configure(text="Loading Whisper STT model (tiny.en)..."))
                self.transcriber = AudioTranscriber(
                    model_size="tiny.en",
                    device_index=default_id,
                    on_transcription_callback=self._on_audio_transcribed,
                    on_level_callback=self._on_audio_level,
                    on_status_callback=self._on_transcriber_status,
                )
                self.after(0, lambda: self.stt_badge.configure(
                    text="STT: Whisper tiny.en (Ready)",
                    fg_color="#064e3b",
                    text_color="#34d399",
                ))
                self.after(0, lambda: self.footer_status.configure(text="Ready to listen"))
            except Exception as e:
                self.after(0, lambda: self.stt_badge.configure(
                    text="STT: Error loading",
                    fg_color="#7f1d1d",
                    text_color="#f87171",
                ))
                self.after(0, lambda: self.footer_status.configure(text=f"Whisper initialization failed: {e}"))

        threading.Thread(target=init_task, daemon=True).start()

    # -------------------------------------------------------------------------
    # Audio & Transcription Callbacks
    # -------------------------------------------------------------------------

    def _on_audio_level(self, rms: float):
        """Updates the live VU activity bar."""
        # Normalize RMS (typical voice 0.01 - 0.2)
        norm_val = min(1.0, rms * 6.0)
        self.after(0, lambda: self.vu_bar.set(norm_val))

    def _on_audio_transcribed(self, text: str, is_final: bool):
        """Appends transcribed speech text in the transcription panel."""
        def update():
            current_text = self.transcription_text.get("1.0", "end-1c").strip()
            # If default placeholder is present, clear it first
            if "Click 'Start Listening'" in current_text:
                self.transcription_text.delete("1.0", "end")
                current_text = ""

            prefix = "\n" if current_text else ""
            self.transcription_text.insert("end", f"{prefix}{text} ")
            self.transcription_text.see("end")

        self.after(0, update)

    def _on_transcriber_status(self, status: str):
        """Updates footer status message."""
        self.after(0, lambda: self.footer_status.configure(text=status))

    # -------------------------------------------------------------------------
    # User Actions (Start, Stop, Suggest Answer, Clear)
    # -------------------------------------------------------------------------

    def on_start_listening(self):
        """User clicked 'Start Listening'."""
        if not self.transcriber:
            messagebox.showwarning("Initializing", "Speech recognition engine is still initializing. Please wait a moment.")
            return

        try:
            self.transcriber.start(device_index=self.selected_device_id)
            self.is_listening = True

            self.start_btn.configure(state="disabled", fg_color="#064e3b")
            self.stop_btn.configure(state="normal", fg_color="#dc2626")
            self.mic_status_dot.configure(text="● RECORDING", text_color="#ef4444")
            self.footer_status.configure(text="Listening for microphone audio...")
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not start microphone: {e}")

    def on_stop_listening(self):
        """User clicked 'Stop Listening'."""
        if not self.transcriber:
            return

        self.transcriber.stop()
        self.is_listening = False

        self.start_btn.configure(state="normal", fg_color="#059669")
        self.stop_btn.configure(state="disabled", fg_color="#7f1d1d")
        self.mic_status_dot.configure(text="● IDLE", text_color="#9ca3af")
        self.vu_bar.set(0.0)
        self.footer_status.configure(text="Stopped listening.")

    def on_suggest_answer(self):
        """User clicked 'Suggest Answer' or pressed Ctrl+Enter."""
        if self.is_generating:
            return

        question = self.transcription_text.get("1.0", "end-1c").strip()
        if not question or "Click 'Start Listening'" in question:
            messagebox.showinfo(
                "No Question",
                "Please speak into the microphone or type/select a question first before requesting a suggestion.",
            )
            return

        # Prepare UI for progressive streaming
        self.is_generating = True
        self.suggest_btn.configure(state="disabled")
        self.stop_gen_btn.configure(state="normal")
        self.stream_status_label.configure(text="Thinking...", text_color="#60a5fa")
        self.stats_label.configure(text="Generating response...")
        self.footer_status.configure(text="Streaming AI answer from local Ollama...")

        self.answer_text.delete("1.0", "end")
        self.full_answer_text = ""
        self.token_count = 0
        self.start_generation_time = time.time()

        selected_mode = self.mode_combo.get()

        # Call local LLM streaming
        self.llm_client.stream_answer(
            question=question,
            mode=selected_mode,
            on_token=self._on_stream_token,
            on_done=self._on_stream_done,
            on_error=self._on_stream_error,
        )

    def _on_stream_token(self, token: str):
        """Progressive streaming token receiver."""
        def append():
            self.token_count += 1
            self.full_answer_text += token
            self.answer_text.insert("end", token)
            self.answer_text.see("end")

            elapsed = max(0.1, time.time() - self.start_generation_time)
            speed = self.token_count / elapsed
            self.stream_status_label.configure(
                text=f"Generating... ({speed:.1f} tok/s)",
                text_color="#34d399",
            )
            self.stats_label.configure(text=f"{self.token_count} tokens | {elapsed:.1f}s")

        self.after(0, append)

    def _on_stream_done(self, full_text: str):
        """Triggered when LLM streaming completes."""
        def finish():
            self.is_generating = False
            self.suggest_btn.configure(state="normal")
            self.stop_gen_btn.configure(state="disabled")
            elapsed = max(0.1, time.time() - self.start_generation_time)
            speed = self.token_count / elapsed
            self.stream_status_label.configure(text="Completed", text_color="#10b981")
            self.stats_label.configure(
                text=f"Finished: {self.token_count} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)"
            )
            self.footer_status.configure(text="Answer generation complete.")

        self.after(0, finish)

    def _on_stream_error(self, err_msg: str):
        """Triggered when LLM generation fails."""
        def handle():
            self.is_generating = False
            self.suggest_btn.configure(state="normal")
            self.stop_gen_btn.configure(state="disabled")
            self.stream_status_label.configure(text="Error", text_color="#ef4444")
            self.answer_text.insert("end", f"\n\n[Generation Error]: {err_msg}\n")
            self.footer_status.configure(text=f"LLM Error: {err_msg}")

        self.after(0, handle)

    def on_stop_generation(self):
        """User clicked 'Stop Generating' to halt output early."""
        self.llm_client.stop_generation()
        self.is_generating = False
        self.suggest_btn.configure(state="normal")
        self.stop_gen_btn.configure(state="disabled")
        self.stream_status_label.configure(text="Halted by user", text_color="#fbbf24")
        self.footer_status.configure(text="Generation stopped.")

    def on_clear_all(self):
        """Clears both transcription and answer text areas."""
        self.transcription_text.delete("1.0", "end")
        self.answer_text.delete("1.0", "end")
        self.stats_label.configure(text="")
        self.stream_status_label.configure(text="Ready", text_color="#9ca3af")
        self.footer_status.configure(text="Cleared.")

    # -------------------------------------------------------------------------
    # Utility Helpers
    # -------------------------------------------------------------------------

    def _toggle_always_on_top(self):
        """Toggles window always-on-top attribute."""
        self.attributes("-topmost", self.pin_var.get())

    def _set_opacity(self, value):
        """Sets window transparency level (stealth mode)."""
        self.attributes("-alpha", float(value))

    def _on_device_changed(self, choice: str):
        """Handles audio input device selection."""
        try:
            device_id = int(choice.split("]")[0].replace("[", "").strip())
            self.selected_device_id = device_id
            if self.is_listening and self.transcriber:
                self.on_stop_listening()
                self.on_start_listening()
        except Exception:
            pass

    def _on_sample_question_selected(self, choice: str):
        """Injects a sample question for immediate evaluation without speaking."""
        match = next((q for q in SAMPLE_QUESTIONS if q["title"] == choice), None)
        if match:
            self.transcription_text.delete("1.0", "end")
            self.transcription_text.insert("1.0", match["text"])
            self.footer_status.configure(text=f"Loaded sample question: {match['title']}")

    def _copy_to_clipboard(self, content: str, toast: str):
        """Copies text to clipboard and updates status."""
        if not content.strip():
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.footer_status.configure(text=f"✓ {toast}")
        # Reset footer after 3 seconds
        self.after(3000, lambda: self.footer_status.configure(text="Ready"))


if __name__ == "__main__":
    app = ParakeetApp()
    app.mainloop()
