from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from core.config import settings


class ModelService:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            dtype=torch.float32,
        )

    def stream_reply(self, messages: list[dict[str, str]]):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            "input_ids": inputs,
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk

    def generate_reply_from_text(self, user_message: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a professional AI travel agent. Only answer travel-related questions.",
            },
            {"role": "user", "content": user_message},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = inputs.shape[1]
        new_tokens = outputs[0][prompt_length:]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return reply.strip()
