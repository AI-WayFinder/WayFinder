import streamlit as st

from agents.chat_orchestrator import ChatOrchestrator
from services.memory_service import MemoryService
from services.model_service import ModelService
from ui.renderers import build_final_response_text, build_streaming_response_html


def handle_user_message(user_input: str) -> None:
    MemoryService.add_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)


def handle_assistant_response(model_service: ModelService) -> None:
    orchestrator = ChatOrchestrator(model_service=model_service)
    user_message = MemoryService.get_latest_user_message()

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        reply = orchestrator.handle(user_message)

        for char in reply:
            full_response += char
            response_placeholder.markdown(
                build_streaming_response_html(full_response),
                unsafe_allow_html=True,
            )

        response_placeholder.markdown(build_final_response_text(full_response))

    MemoryService.add_message("assistant", full_response)
