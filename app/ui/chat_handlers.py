import streamlit as st

from services.memory_service import MemoryService
from services.model_service import ModelService
from ui.renderers import build_final_response_text, build_streaming_response_html


def handle_user_message(user_input: str) -> None:
    MemoryService.add_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)


def handle_assistant_response(model_service: ModelService) -> None:
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in model_service.stream_reply(MemoryService.get_model_messages()):
            full_response += chunk

            response_placeholder.markdown(
                build_streaming_response_html(full_response),
                unsafe_allow_html=True,
            )

        response_placeholder.markdown(build_final_response_text(full_response))

    MemoryService.add_message("assistant", full_response)
