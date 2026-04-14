"""Data model for chat messages exchanged between user, assistant, and system."""

from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }
