"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено."

SYSTEM_PROMPT = (
    "Ты ассистент диспетчера сервисной службы «ПолярТех»."
    "Отвечай только по результатам инструментов."
    "Если номер детали неизвестен или нужно проверить её замену по бюллетеню, сначала вызови search_knowledge."
    "Для обзорных вопросов по линейкам вызывай summarize_topic."
    "В ответе называй все найденные факты: узел, актуальный код детали, остаток в шт. и цену в руб."
)

REQUIRED_ARGS = {s["function"]["name"]: set(s["function"]["parameters"]["required"]) for s in TOOL_SCHEMAS}

# паттерн цен и остатков
_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\w-])"
_WITH_UNIT = re.compile(rf"({_NUMBER})\s*(руб|₽|шт)", re.IGNORECASE)

def _known_codes(question: str, steps: list[ToolStep]) -> set[str]:
    """Все коды из вопроса и успешных результатов прошлых шагов."""
    text = question + " " + json.dumps([s.result for s in steps if s.result], ensure_ascii=False)
    return set(find_codes(text))


def _check_call(call: ToolCall, tools: dict, question: str, steps: list[ToolStep]) -> str | None:
    """Проверка имени инструмента, аргументов, номера детали и количества в заказе (FR-7)."""
    if call.name not in tools:
        return f"Неизвестный инструмент: {call.name}"
    if not isinstance(call.arguments, dict) or "_raw" in call.arguments:
        return "Некорректный JSON в аргументах вызова"
    required = REQUIRED_ARGS.get(call.name, set())
    if not required.issubset(call.arguments):
        return f"Не хватает обязательных аргументов: {sorted(required - set(call.arguments))}"

    if "part_id" in call.arguments:
        part_id = str(call.arguments["part_id"])
        known_parts = {c for c in _known_codes(question, steps) if code_type(c) == "Part"}
        if part_id not in known_parts:
            return (
                f"Деталь {part_id} не встречалась в вопросе или результатах. "
                "Сначала вызовите search_knowledge."
            )

    if call.name == "create_service_order":
        qty = call.arguments.get("qty")
        # исключение кодов из вопроса и проверка называл ли пользователь число qty
        q_without_codes = re.sub(r"[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,4}[A-Za-zА-Яа-яЁё]?", " ", question)
        user_nums = {int(n) for n in re.findall(r"\d+", q_without_codes)}
        if not isinstance(qty, int) or qty not in user_nums:
            return "Количество не указано пользователем в вопросе. Уточните количество у пользователя."

    return None


def _has_hallucinations(answer: str, question: str, steps: list[ToolStep]) -> bool:
    """Сверка кодов, цен и остатков в ответе с вопросом и результатами инструментов (FR-10)."""
    known = _known_codes(question, steps)
    if any(code not in known for code in find_codes(answer)):
        return True
    tool_text = json.dumps([s.result for s in steps if s.result], ensure_ascii=False)
    allowed_nums = set(re.findall(r"\d+", tool_text)) | set(re.findall(r"\d+", question))
    for raw_num, _ in _WITH_UNIT.findall(answer):
        digits = re.sub(r"\D", "", raw_num)
        if digits not in allowed_nums:
            return True
    return False


def _sources(steps: list[ToolStep]) -> list[str]:
    found: list[str] = []
    for step in steps:
        for src in (step.result or {}).get("sources", []):
            if src not in found:
                found.append(src)
    return found


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    seen_calls: dict[tuple[str, str], dict] = {}
    answer = ""

    while len(steps) < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())

        if not response.tool_calls:
            answer = response.content.strip()
            break

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                break

            err = _check_call(call, tools, question, steps)
            if err:
                steps.append(ToolStep(call.name, dict(call.arguments), error=err))
                messages.append(tool_message(call, {"error": err}))
                continue

            if call.name == "create_service_order" and not confirm:
                pending = {"tool": call.name, "args": dict(call.arguments)}
                messages.append(tool_message(call, {"status": "waiting_confirmation", **call.arguments}))
                reply = llm.chat(messages).content.strip() or "Подтвердите создание заказа."
                return AgentResult(
                    answer=reply,
                    steps=steps,
                    sources=_sources(steps),
                    needs_confirmation=True,
                    pending_action=pending,
                )

            key = (call.name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
            if key in seen_calls:
                messages.append(tool_message(call, seen_calls[key]))
                continue

            result = None
            last_exc = None
            for _ in range(3):
                try:
                    result = tools[call.name](**call.arguments)
                    last_exc = None
                    break
                except ErpUnavailable as exc:
                    last_exc = exc

            if last_exc is not None:
                err_msg = f"Данные ERP получить не удалось: {last_exc}"
                steps.append(ToolStep(call.name, dict(call.arguments), error=err_msg))
                messages.append(tool_message(call, {"error": err_msg}))
                continue

            assert result is not None
            seen_calls[key] = result
            step_err = result.get("error") if isinstance(result, dict) else None
            steps.append(ToolStep(call.name, dict(call.arguments), result=result, error=step_err))
            messages.append(tool_message(call, result))
    else:
        messages.append({"role": "user", "content": "Лимит шагов исчерпан. Подведи итог по тому, что удалось узнать."})
        answer = llm.chat(messages).content.strip() or "Работа прервана: исчерпан лимит шагов."

    sources = _sources(steps)
    kb_refused = any((s.result or {}).get("refused") for s in steps)
    erp_not_found = any("не найдена в ERP" in (s.error or "") for s in steps)

    if "в базе знаний не найдено" in answer.lower() or kb_refused or erp_not_found:
        return AgentResult(answer=REFUSAL, steps=steps, sources=sources, refused=True)

    if _has_hallucinations(answer, question, steps):
        fix_prompt = "В ответе есть выдуманные коды, цены или остатки. Исправь ответ строго по данным инструментов."
        messages.append({"role": "user", "content": fix_prompt})
        answer = llm.chat(messages).content.strip()
        if _has_hallucinations(answer, question, steps) or "в базе знаний не найдено" in answer.lower():
            return AgentResult(answer=REFUSAL, steps=steps, sources=sources, refused=True)

    return AgentResult(answer=answer, steps=steps, sources=sources, refused=False)
