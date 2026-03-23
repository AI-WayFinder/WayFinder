from typing import List

import streamlit as st

from models.chat import ChatMessage
from prompts.system_prompts import TRAVEL_AGENT_SYSTEM_PROMPT


class MemoryService:
    SESSION_KEY = "messages"

    @classmethod
    def initialize(cls) -> None:
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = [
                ChatMessage(
                    role="system",
                    content=TRAVEL_AGENT_SYSTEM_PROMPT,
                ),
                ChatMessage(
                    role="assistant",
                    content="Hi. I’m WayFinder, your travel planning assistant. Where would you like to go?",
                ),
            ]

    @classmethod
    def get_messages(cls) -> List[ChatMessage]:
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def add_message(cls, role: str, content: str) -> None:
        st.session_state[cls.SESSION_KEY].append(
            ChatMessage(role=role, content=content)
        )

    @classmethod
    def get_model_messages(cls, max_history: int = 8) -> list[dict[str, str]]:
        messages = st.session_state[cls.SESSION_KEY]

        system_message = messages[0]
        recent_messages = messages[1:][-max_history:]

        return [system_message.to_dict()] + [msg.to_dict() for msg in recent_messages]

    @classmethod
    def get_display_messages(cls) -> List[ChatMessage]:
        return [
            msg for msg in st.session_state[cls.SESSION_KEY] if msg.role != "system"
        ]

    @classmethod
    def clear(cls) -> None:
        st.session_state[cls.SESSION_KEY] = [
            ChatMessage(
                role="system",
                content=TRAVEL_AGENT_SYSTEM_PROMPT,
            ),
            ChatMessage(
                role="assistant",
                content="Hi. I’m WayFinder, your travel planning assistant. Where would you like to go?",
            ),
        ]

    @classmethod
    def get_latest_user_message(cls) -> str:
        for message in reversed(st.session_state[cls.SESSION_KEY]):
            if message.role == "user":
                return message.content
        return ""
