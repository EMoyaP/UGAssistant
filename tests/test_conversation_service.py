from __future__ import annotations

import asyncio
import unittest

from ugassistant.adapters.simulated import SimulatedLLMAdapter
from ugassistant.services.conversation import ConversationService


class ConversationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_answers_in_detected_language_with_limited_history(self) -> None:
        adapter = SimulatedLLMAdapter(response="Una respuesta local muy breve.")
        service = ConversationService(
            adapter,
            inference_lock=asyncio.Lock(),
            max_history_turns=1,
        )

        first = await service.answer("Como estas?", "es")
        second = await service.answer("Et maintenant?", "fr")

        self.assertEqual(first, "Una respuesta local muy breve.")
        self.assertEqual(second, "Una respuesta local muy breve.")
        self.assertIn("francais", adapter.messages[-1][0].content)
        self.assertEqual(len(adapter.messages[-1]), 4)
        self.assertEqual(service.status.language, "fr")

    async def test_rejects_languages_other_than_spanish_and_french(self) -> None:
        service = ConversationService(
            SimulatedLLMAdapter(),
            inference_lock=asyncio.Lock(),
        )

        with self.assertRaises(ValueError):
            await service.answer("Hello", "en")

    async def test_complete_response_uses_a_detailed_prompt_and_larger_limit(self) -> None:
        response = " ".join(["explicacion"] * 90)
        adapter = SimulatedLLMAdapter(response=response)
        service = ConversationService(
            adapter,
            inference_lock=asyncio.Lock(),
            max_response_characters=80,
            max_tokens=32,
            complete_max_response_characters=1200,
            complete_max_tokens=384,
        )

        answer = await service.answer("Explica el proceso", "es", "complete")

        self.assertEqual(answer, response)
        self.assertIn("respuesta completa", adapter.messages[-1][0].content)

    async def test_rejects_unknown_response_detail(self) -> None:
        service = ConversationService(
            SimulatedLLMAdapter(),
            inference_lock=asyncio.Lock(),
        )

        with self.assertRaises(ValueError):
            await service.answer("Explica el proceso", "es", "extended")


if __name__ == "__main__":
    unittest.main()
