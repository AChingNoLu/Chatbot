"""Groq JSON generation; credentials come exclusively from a local .env file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


class GroqClient:
    def __init__(self, env_path: Path | None = None) -> None:
        path = (
            env_path
            if env_path is not None
            else Path(__file__).resolve().parents[1] / ".env"
        )

        if not path.is_file():
            raise ValueError(
                "Thiếu file .env ở thư mục project; "
                "thêm GROQ_API_KEY vào file này."
            )

        values = dotenv_values(
            path,
            encoding="utf-8",
            interpolate=False,
        )

        key = (values.get("GROQ_API_KEY") or "").strip()

        if not key or "${" in key:
            raise ValueError(
                "File .env phải chứa GROQ_API_KEY hợp lệ."
            )

        self._api_key = key

        self.model = (
            values.get("GROQ_MODEL")
            or "openai/gpt-oss-20b"
        ).strip()

        self._client = None

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str,
        schema: dict,
    ) -> Any:

        from groq import Groq

        try:
            if self._client is None:
                self._client = Groq(
                    api_key=self._api_key
                )

            response = self._client.chat.completions.create(
                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                temperature=0,

                # Không cần model "suy nghĩ" nhiều
                # vì RAG chỉ cần chọn evidence trong context.
                reasoning_effort="low",

                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "legal_rag_response",

                        # RESPONSE_SCHEMA của bạn phù hợp strict mode.
                        "strict": True,

                        "schema": schema,
                    },
                },
            )

        except Exception as e:
            error_message = str(e).replace(
                self._api_key,
                "[REDACTED]"
            )

            print("\n========== GROQ ERROR ==========")
            print("Type:", type(e).__name__)
            print(
                "Status:",
                getattr(e, "status_code", None)
            )
            print(
                "Code:",
                getattr(e, "code", None)
            )
            print("Message:", error_message)
            print("Model:", self.model)
            print("================================\n")

            raise RuntimeError(
                "Groq request failed. "
                "Xem GROQ ERROR phía trên."
            ) from None

        try:
            content = (
                response.choices[0]
                .message
                .content
            )

            if not content:
                return None

            return json.loads(content)

        except (ValueError, TypeError, IndexError, AttributeError):
            return None

    def close(self) -> None:
        # Groq client không bắt buộc phải close như Gemini client.
        self._client = None