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

        from groq import DefaultHttpxClient, Groq

        try:
            if self._client is None:
                self._client = Groq(
                    api_key=self._api_key,
                    http_client=DefaultHttpxClient(trust_env=False),
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

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Return the assistant message, including structured tool calls if any.

        Callers must execute requested tools themselves and append tool results
        to the message history before requesting the next assistant step.
        """
        if not messages or not all(isinstance(item, dict) for item in messages):
            raise ValueError("messages must be a non-empty list of message objects")
        if not isinstance(tools, list) or not tools:
            raise ValueError("tools must be a non-empty list of tool definitions")

        from groq import DefaultHttpxClient, Groq

        try:
            if self._client is None:
                self._client = Groq(
                    api_key=self._api_key,
                    http_client=DefaultHttpxClient(trust_env=False),
                )
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0,
            )
            message = response.choices[0].message
            calls = []
            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid tool arguments for {call.function.name}"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"Tool arguments for {call.function.name} must be an object"
                    )
                calls.append({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": arguments,
                    },
                })
            return {"role": "assistant", "content": message.content, "tool_calls": calls}
        except ValueError:
            raise
        except Exception as exc:
            safe_error = str(exc).replace(self._api_key, "[REDACTED]")
            raise RuntimeError(
                f"Groq tool calling failed ({type(exc).__name__}): {safe_error}"
            ) from None

    def close(self) -> None:
        # Groq client không bắt buộc phải close như Gemini client.
        if self._client is not None:
            self._client.close()
            self._client = None
