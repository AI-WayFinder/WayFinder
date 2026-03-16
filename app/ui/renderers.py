import html


def build_streaming_response_html(text: str) -> str:
    safe_text = html.escape(text)

    return f"""
    <div class="streaming-response">
        {safe_text}<span class="blinking-cursor">▌</span>
    </div>
    """


def build_final_response_text(text: str) -> str:
    return text
