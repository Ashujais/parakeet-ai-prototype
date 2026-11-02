"""
Local LLM Client for Parakeet AI Prototype
Connects to local Ollama instance and streams progressive answer generation.
"""

import json
import urllib.request
import urllib.error
import threading
from typing import Callable, Optional, List, Dict


class LocalLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5:3b",
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._current_request_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()

    def check_health(self) -> Dict[str, any]:
        """Checks if Ollama server is running and returns available models."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                headers={"User-Agent": "ParakeetAI-Desktop/1.0"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                return {
                    "connected": True,
                    "models": models,
                    "current_model": (
                        self.default_model if self.default_model in models
                        else (models[0] if models else self.default_model)
                    ),
                }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "models": [],
                "current_model": self.default_model,
            }

    def get_system_prompt(self, mode: str) -> str:
        """Returns specialized system prompts inspired by Parakeet AI copilots."""
        if mode == "Concise Bullet Points":
            return (
                "You are an AI meeting and interview copilot inspired by Parakeet AI. "
                "Provide a rapid, bullet-pointed answer highlighting the most important "
                "points, key decisions, and critical takeaways. Be concise, punchy, and direct."
            )
        elif mode == "Technical Deep Dive":
            return (
                "You are an expert technical advisor and staff software engineer copilot. "
                "Provide a thorough, high-level technical response detailing architecture, "
                "trade-offs, algorithmic complexity, performance bottlenecks, and best practices. "
                "Use clean markdown formatting and code examples where appropriate."
            )
        else:  # Default: Direct Interview Answer
            return (
                "You are a real-time interview and meeting copilot inspired by Parakeet AI. "
                "The user is speaking or being asked a question. Formulate a confident, articulate, "
                "structured, and professional answer as if speaking in an interview or business meeting. "
                "Start directly with the core answer. Use structured formatting for readability. "
                "Avoid meta-commentary like 'Certainly! Here is an answer:'."
            )

    def stream_answer(
        self,
        question: str,
        mode: str = "Direct Interview Answer",
        model: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """Spawns a background thread that streams generation tokens from Ollama."""
        self.stop_generation()
        self._stop_requested.clear()

        selected_model = model or self.default_model
        system_prompt = self.get_system_prompt(mode)

        def worker():
            full_response = []
            try:
                payload = {
                    "model": selected_model,
                    "prompt": f"{system_prompt}\n\nQuestion / Context:\n{question.strip()}\n\nAnswer:",
                    "stream": True,
                    "options": {
                        "temperature": 0.6,
                        "top_p": 0.9,
                    },
                }

                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "ParakeetAI-Desktop/1.0",
                    },
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw_line in resp:
                        if self._stop_requested.is_set():
                            break
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        chunk_obj = json.loads(line)
                        token = chunk_obj.get("response", "")
                        if token:
                            full_response.append(token)
                            if on_token:
                                on_token(token)

                        if chunk_obj.get("done", False):
                            break

                final_text = "".join(full_response)
                if on_done:
                    on_done(final_text)

            except Exception as e:
                if not self._stop_requested.is_set():
                    if on_error:
                        on_error(f"Ollama connection error: {e}")

        self._current_request_thread = threading.Thread(target=worker, daemon=True)
        self._current_request_thread.start()

    def stop_generation(self):
        """Signals to cancel currently active streaming generation."""
        self._stop_requested.set()
