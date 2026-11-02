"""
Test Audio & Sample Data Generator for Parakeet AI Prototype
Provides sample interview and meeting questions and test audio synthesis.
"""

import numpy as np

SAMPLE_QUESTIONS = [
    {
        "category": "System Design",
        "title": "Database Indexing & B-Trees",
        "text": "Can you explain how database indexes work under the hood, specifically why B-Trees are preferred over binary search trees for disk storage?",
    },
    {
        "category": "Behavioral / Leadership",
        "title": "Handling Disagreements",
        "text": "Tell me about a time when you strongly disagreed with a senior engineer or product manager on a technical design decision. How did you resolve it?",
    },
    {
        "category": "Distributed Systems",
        "title": "CAP Theorem & Eventual Consistency",
        "text": "How do you handle eventual consistency in a distributed microservices architecture when an immediate read-after-write is required by the user?",
    },
    {
        "category": "Python / Concurrency",
        "title": "GIL & Asyncio vs Multiprocessing",
        "text": "What is the Python Global Interpreter Lock (GIL), and when would you choose asyncio versus multiprocessing for a high-throughput backend service?",
    },
    {
        "category": "General Engineering",
        "title": "Caching Strategy & Invalidation",
        "text": "What caching strategies do you use for high-traffic read APIs, and how do you handle cache invalidation and the thundering herd problem?",
    },
]


def generate_synthetic_tone(duration_sec: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates a pleasant synthetic tone burst simulating audio activity."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Fundamental frequency + harmonics with smooth attack and decay envelope
    envelope = np.sin(np.pi * t / duration_sec) ** 2
    tone = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1320 * t)
    ) * envelope
    return tone.astype(np.float32)
