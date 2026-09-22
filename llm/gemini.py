"""Gemini JSON generation; credentials come exclusively from a local .env file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


class GeminiClient:
    def __init__(self, env_path: Path | None = None) -> None:
        path = env_path if env_path is not None else Path(
            __file__).resolve().parents[1] / ".env"
        if not path.is_file():
            raise ValueError(
                "Thiếu file .env ở thư mục project; thêm GEMINI_API_KEY vào file này.")
        # No load_dotenv/os.getenv: pre-existing process environment cannot supply the key.
        values = dotenv_values(path, encoding="utf-8", interpolate=False)
        key = (values.get("GEMINI_API_KEY") or values.get(
            "GOOGLE_API_KEY") or "").strip()
        if not key or "${" in key:
            raise ValueError(
                "File .env phải chứa GEMINI_API_KEY hoặc GOOGLE_API_KEY hợp lệ.")
        self._api_key = key
        self.model = (values.get("GEMINI_MODEL") or "gemini-3.6-flash").strip()
        self._client = None

    def generate(self, prompt: str, *, system_instruction: str, schema: dict) -> Any:
        from google import genai
        from google.genai import types

        try:
            if self._client is None:
                self._client = genai.Client(
                    api_key=self._api_key,
                    vertexai=False
                )

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=schema,

                    # Có thể thêm dòng này để tránh AFC:
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )

        except Exception as e:
            # Che API key nếu SDK vô tình đưa nó vào message.
            error_message = str(e).replace(self._api_key, "[REDACTED]")

            print("\n========== GEMINI ERROR ==========")
            print("Type:", type(e).__name__)
            print("Status:", getattr(e, "status_code", None))
            print("Code:", getattr(e, "code", None))
            print("Message:", error_message)
            print("Model:", self.model)
            print("==================================\n")

            raise RuntimeError(
                "Gemini request failed. Xem GEMINI ERROR phía trên."
            ) from None

        try:
            # SDK có thể parse structured output trực tiếp.
            if getattr(response, "parsed", None) is not None:
                return response.parsed

            return json.loads(response.text or "")

        except (ValueError, TypeError):
            return None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
