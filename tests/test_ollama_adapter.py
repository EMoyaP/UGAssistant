from __future__ import annotations

import unittest

from ugassistant.adapters.ollama import OllamaAdapter
from ugassistant.domain.ports import LLMMessage


class OllamaAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_checks_the_exact_locked_model_and_sends_non_streaming_chat(self) -> None:
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            model="gemma3:4b",
        )
        requests: list[tuple[str, dict[str, object] | None]] = []

        def request(path: str, payload: dict[str, object] | None) -> dict[str, object]:
            requests.append((path, payload))
            if path == "/api/tags":
                return {"models": [{"name": "gemma3:4b"}]}
            return {"message": {"content": "Respuesta local."}}

        adapter._request = request  # type: ignore[method-assign]

        status = await adapter.status()
        answer = await adapter.chat(
            (LLMMessage("user", "Hola"),),
            max_tokens=80,
            temperature=0.2,
            repeat_penalty=1.08,
            think=False,
            context_tokens=4096,
        )

        self.assertTrue(status.available)
        self.assertTrue(status.model_available)
        self.assertEqual(answer, "Respuesta local.")
        self.assertEqual(requests[1][0], "/api/chat")
        payload = requests[1][1]
        self.assertEqual(payload["model"], "gemma3:4b")  # type: ignore[index]
        self.assertFalse(payload["stream"])  # type: ignore[index]
        self.assertFalse(payload["think"])  # type: ignore[index]
        self.assertEqual(payload["options"]["num_ctx"], 4096)  # type: ignore[index]
        self.assertEqual(payload["options"]["repeat_penalty"], 1.08)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
