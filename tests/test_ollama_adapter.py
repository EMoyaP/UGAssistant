from __future__ import annotations

import unittest

from ugassistant.adapters.ollama import OllamaAdapter
from ugassistant.domain.ports import LLMMessage


class OllamaAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_checks_the_exact_locked_model_and_sends_non_streaming_chat(self) -> None:
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            model="qwen3:4b-instruct",
        )
        requests: list[tuple[str, dict[str, object] | None]] = []

        def request(path: str, payload: dict[str, object] | None) -> dict[str, object]:
            requests.append((path, payload))
            if path == "/api/tags":
                return {"models": [{"name": "qwen3:4b-instruct"}]}
            return {"message": {"content": "Respuesta local."}}

        adapter._request = request  # type: ignore[method-assign]

        status = await adapter.status()
        answer = await adapter.chat(
            (LLMMessage("user", "Hola"),),
            max_tokens=80,
            temperature=0.2,
            think=True,
            context_tokens=4096,
        )

        self.assertTrue(status.available)
        self.assertTrue(status.model_available)
        self.assertEqual(answer, "Respuesta local.")
        self.assertEqual(requests[1][0], "/api/chat")
        payload = requests[1][1]
        self.assertEqual(payload["model"], "qwen3:4b-instruct")  # type: ignore[index]
        self.assertFalse(payload["stream"])  # type: ignore[index]
        self.assertTrue(payload["think"])  # type: ignore[index]
        self.assertEqual(payload["options"]["num_ctx"], 4096)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
