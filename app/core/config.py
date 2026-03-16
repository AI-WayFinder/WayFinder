from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_title: str = "WayFinder: Your Travel Planning Assistant"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 200


settings = Settings()
