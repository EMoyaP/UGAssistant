from __future__ import annotations

import asyncio
import unittest

from ugassistant.adapters.simulated import SimulatedLLMAdapter
from ugassistant.domain.ports import LLMMessage
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

    async def test_complete_response_uses_larger_context_without_thinking_mode(self) -> None:
        response = " ".join(["explicacion"] * 90)
        adapter = SimulatedLLMAdapter(response=response)
        service = ConversationService(
            adapter,
            inference_lock=asyncio.Lock(),
            short_context_tokens=2048,
            complete_context_tokens=4096,
            max_tokens=32,
            complete_max_tokens=384,
        )

        answer = await service.answer("Explica el proceso", "es", "complete")

        self.assertEqual(answer, response)
        self.assertIn("respuesta completa", adapter.messages[-1][0].content)
        self.assertEqual(adapter.thinking_modes, [False])
        self.assertEqual(adapter.context_windows, [4096])

    async def test_short_response_uses_short_context_and_precision_policy(self) -> None:
        adapter = SimulatedLLMAdapter(response="Respuesta breve.")
        service = ConversationService(
            adapter,
            inference_lock=asyncio.Lock(),
            short_context_tokens=2048,
            complete_context_tokens=4096,
        )

        await service.answer("Resume esto", "es", "short")

        self.assertEqual(adapter.thinking_modes, [False])
        self.assertEqual(adapter.context_windows, [2048])
        self.assertEqual(adapter.repeat_penalties, [1.08])
        self.assertIn("tres frases", adapter.messages[-1][0].content)
        self.assertIn("omite los detalles inciertos", adapter.messages[-1][0].content)

    def test_complete_prompt_forbids_unfounded_certainty(self) -> None:
        prompt = ConversationService._system_prompt("es", "complete")

        self.assertIn("autentico", prompt)
        self.assertIn("Omite los detalles inciertos", prompt)
        self.assertIn("no puedes verificar", prompt)

    async def test_rejects_unknown_response_detail(self) -> None:
        service = ConversationService(
            SimulatedLLMAdapter(),
            inference_lock=asyncio.Lock(),
        )

        with self.assertRaises(ValueError):
            await service.answer("Explica el proceso", "es", "extended")

    async def test_removes_markdown_before_storing_or_speaking_a_response(self) -> None:
        adapter = SimulatedLLMAdapter(
            response=(
                "### **Gaspacho andaluz**\n"
                "- Tomate y aceite.\n"
                "1. Tritura los ingredientes.\n"
                "[Fuente](https://example.invalid)"
            )
        )
        service = ConversationService(adapter, inference_lock=asyncio.Lock())

        answer = await service.answer("Dame una receta", "es", "complete")

        self.assertIn("Gaspacho andaluz", answer)
        self.assertIn("Tomate y aceite.", answer)
        self.assertIn("1. Tritura los ingredientes.", answer)
        self.assertNotRegex(answer, r"[#*\[\]`]")

    async def test_keeps_history_within_the_context_token_budget(self) -> None:
        adapter = SimulatedLLMAdapter(response="Respuesta local.")
        service = ConversationService(
            adapter,
            inference_lock=asyncio.Lock(),
            max_history_turns=3,
            short_context_tokens=1024,
            max_tokens=256,
        )
        service._history = [  # type: ignore[attr-defined]
            LLMMessage("user", "x" * 1600),
            LLMMessage("assistant", "y" * 1600),
            LLMMessage("user", "z" * 1600),
            LLMMessage("assistant", "w" * 1600),
        ]

        await service.answer("Pregunta breve", "es")

        messages = adapter.messages[-1]
        estimated_tokens = sum(
            service._estimated_tokens(message.content)  # type: ignore[attr-defined]
            for message in messages
        )
        self.assertLessEqual(estimated_tokens, 1024 - 256 - 128)


if __name__ == "__main__":
    unittest.main()
