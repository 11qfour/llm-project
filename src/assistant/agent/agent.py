"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS, WAREHOUSES
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import LOOKALIKES, code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено."

SYSTEM_PROMPT = (
    "Ты ассистент диспетчера сервисной службы «ПолярТех». Отвечай только по результатам инструментов. "
    "Для конкретных моделей, ошибок, деталей, поставщиков и регламентов вызывай search_knowledge. "
    "Передавай в поиск полный вопрос пользователя: не теряй просьбу назвать поставщика или модели. "
    "Для обзорных вопросов по линейкам и статистике отказов вызывай summarize_topic. "
    "Если нужна актуальная деталь, пройди всю цепочку замен из facts и используй последнюю. "
    "Цену получай через get_price, остаток через check_stock. "
    "Для оформления заказа обязательно вызови create_service_order с моделью, деталью, складом и количеством. "
    "Проверка цены и остатка не создаёт заказ. Не подменяй вызов инструмента текстом об оформлении. "
    "При waiting_confirmation попроси подтверждение: заказ ещё не создан. "
    "Отвечай на заданный вопрос. Не добавляй цену и остаток, если их не спрашивали."
)

REQUIRED_ARGS = {s["function"]["name"]: set(s["function"]["parameters"]["required"]) for s in TOOL_SCHEMAS}

# паттерн цен и остатков
_CODE_RE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])"
    r"(?:SO-\d+|(?:SB|R)-\d{4}-\d{2,3}|(?:AR|BZ|GS|SP|P|E)[- ]?\d{2,4}[A-Z]?)"
    r"(?![0-9A-Za-zА-Яа-яЁё])"
)
_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\w-])"
_WITH_UNIT = re.compile(rf"({_NUMBER})\s*(руб|₽|шт)", re.IGNORECASE)
_ORDER_REQUEST = re.compile(
    r"\bзакажи(?:те)?\b|\b(?:оформи(?:те)?|создай(?:те)?|подготовь(?:те)?)\b[^.!?\n]*\bзаказ\b",
    re.IGNORECASE,
)


def _known_codes(question: str, steps: list[ToolStep]) -> set[str]:
    """Все коды из вопроса и успешных результатов прошлых шагов."""
    text = question + " " + json.dumps([s.result for s in steps if s.result], ensure_ascii=False)
    return set(find_codes(text))


def _is_code_token(token: str) -> bool:
    if token.startswith(("SO-", "SB-", "R-")):
        return True
    return normalize_code(token) is not None


def _numbers_without_codes(text: str) -> set[str]:
    """Все числа из текста после удаления артикулов, кодов моделей, бюллетеней и заказов."""
    text = text.upper().translate(LOOKALIKES)
    cleaned = _CODE_RE.sub(lambda m: " " if _is_code_token(m.group(0)) else m.group(0), text)
    return set(re.findall(r"\d+", cleaned))


def _check_call(call: ToolCall, tools: dict, question: str, steps: list[ToolStep]) -> str | None:
    """Проверка имени инструмента, аргументов, номера детали и количества в заказе (FR-7)."""
    if call.name not in tools:
        return f"Неизвестный инструмент: {call.name}"
    if not isinstance(call.arguments, dict) or "_raw" in call.arguments:
        return "Некорректный JSON в аргументах вызова"
    required = REQUIRED_ARGS.get(call.name, set())
    if set(call.arguments) != required:
        return f"Некорректный набор аргументов: ожидаются {sorted(required)}"

    if "warehouse" in call.arguments and call.arguments["warehouse"] not in WAREHOUSES:
        return f"Неизвестный склад: {call.arguments['warehouse']}"

    if "part_id" in call.arguments:
        part_id = str(call.arguments["part_id"])
        known_parts = {c for c in _known_codes(question, steps) if code_type(c) == "Part"}
        if part_id not in known_parts:
            return f"Деталь {part_id} не встречалась в вопросе или результатах. Сначала вызовите search_knowledge."

    if call.name == "create_service_order":
        model = call.arguments.get("model")
        norm_model = normalize_code(model) if isinstance(model, str) else None
        if not norm_model or code_type(norm_model) != "Model":
            return f"Некорректный код модели: {model}"
        if norm_model not in _known_codes(question, steps):
            return "Модель заказа не встречалась в вопросе или результатах инструментов. Уточните модель."
        qty = call.arguments.get("qty")
        user_nums = {int(n) for n in _numbers_without_codes(question)}
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0 or qty not in user_nums:
            return "Количество не указано пользователем в вопросе. Уточните количество у пользователя."

    return None


def _has_hallucinations(answer: str, question: str, steps: list[ToolStep]) -> bool:
    """Сверка кодов, цен и остатков в ответе с вопросом и результатами инструментов (FR-10)."""
    known = _known_codes(question, steps)
    if any(code not in known for code in find_codes(answer)):
        return True
    tool_results = [{k: v for k, v in s.result.items() if k != "sources"} for s in steps if s.result and not s.error]
    tool_text = json.dumps(tool_results, ensure_ascii=False)
    allowed_nums = _numbers_without_codes(tool_text) | _numbers_without_codes(question)
    for raw_num, _ in _WITH_UNIT.findall(answer):
        digits = re.sub(r"\D", "", raw_num)
        if digits not in allowed_nums:
            return True
    return False


def _missing_replacements(answer: str, question: str, steps: list[ToolStep]) -> dict[str, list[str]]:
    """Новый для пользователя снятый артикул нельзя назвать без известной замены."""
    replacements: dict[str, set[str]] = {}
    for step in steps:
        if step.tool != "search_knowledge" or step.error or not step.result or step.result.get("refused"):
            continue
        for fact in step.result.get("facts", []):
            codes = find_codes(fact)
            if " -REPLACED_BY-> " in fact and len(codes) == 2 and all(code_type(c) == "Part" for c in codes):
                replacements.setdefault(codes[0], set()).add(codes[1])

    answer_codes = set(find_codes(answer))
    missing = {}
    for old in sorted(answer_codes - set(find_codes(question))):
        visited = {old}
        pending = list(replacements.get(old, []))
        while pending:
            new = pending.pop()
            if new not in visited:
                visited.add(new)
                pending.extend(replacements.get(new, []))
        successors = visited - {old}
        if successors and not answer_codes.intersection(successors):
            missing[old] = sorted(successors)
    return missing


def _sources(steps: list[ToolStep]) -> list[str]:
    found: list[str] = []
    for step in steps:
        for src in (step.result or {}).get("sources", []):
            if src not in found:
                found.append(src)
    return found


def _confirmation_text(arguments: dict) -> str:
    """Статус заказа формируется кодом из проверенных аргументов, а не моделью."""
    return (
        f"Подтвердите создание заказа: деталь {arguments['part_id']}, "
        f"количество: {arguments['qty']}, модель {arguments['model']}, склад {arguments['warehouse']}. "
        "Заказ ещё не создан."
    )


def _requests_order(question: str) -> bool:
    """Распознаёт явную команду заказать; отрицание перед командой её отменяет."""
    return any(
        not re.search(r"\bне\s*$", question[: match.start()], re.IGNORECASE)
        for match in _ORDER_REQUEST.finditer(question)
    )


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
    pending_action: dict | None = None
    created: ToolStep | None = None
    answer = ""
    order_requested = _requests_order(question)
    order_reminded = False

    while len(steps) < max_steps and pending_action is None and created is None:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())

        if not response.tool_calls:
            # Один дополнительный шанс выбрать инструмент, если модель завершила заказ обычным текстом.
            order_attempted = any(step.tool == "create_service_order" for step in steps)
            has_data = any(step.result and not step.error and not step.result.get("refused") for step in steps)
            if order_requested and has_data and not order_attempted and not order_reminded:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Заказ ещё не создан и не передан на подтверждение. "
                            "Для запрошенного заказа вызови create_service_order. "
                            "Если не хватает аргументов, уточни их; не сообщай об успешном оформлении."
                        ),
                    }
                )
                order_reminded = True
                continue
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
                pending_action = {"tool": call.name, "args": dict(call.arguments)}
                answer = _confirmation_text(call.arguments)
                break

            key = (call.name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
            if key in seen_calls:
                cached = seen_calls[key]
                steps.append(ToolStep(call.name, dict(call.arguments), result=cached, error=cached.get("error")))
                messages.append(tool_message(call, cached))
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
            if call.name == "create_service_order" and not step_err and result.get("order_id"):
                created = steps[-1]
                answer = f"Заказ {result['order_id']} создан. Деталь {call.arguments['part_id']}."
                break

    if not answer and pending_action is None:
        prompt = "Лимит шагов исчерпан. Подведи итог по тому, что удалось узнать."
        messages.append({"role": "user", "content": prompt})
        answer = llm.chat(messages).content.strip() or "Работа прервана: исчерпан лимит шагов."

    sources = _sources(steps)
    has_valid_data = any(s.result and not s.error and not s.result.get("refused") for s in steps)
    kb_refused = any((s.result or {}).get("refused") for s in steps)
    erp_not_found = any("не найдена в ERP" in (s.error or "") for s in steps)
    if pending_action is None and not has_valid_data and (kb_refused or erp_not_found):
        return AgentResult(answer=REFUSAL, steps=steps, sources=sources, refused=True)
    if pending_action is not None:
        return AgentResult(
            answer=answer,
            steps=steps,
            sources=sources,
            needs_confirmation=True,
            pending_action=pending_action,
        )

    if created is not None:
        answer = f"Заказ {created.result['order_id']} создан. Деталь {created.args['part_id']}."
    elif order_requested:
        # Даже после неудачного напоминания нельзя отдавать неподтверждённое «заказ оформлен».
        errors = [step.error for step in steps if step.error]
        detail = errors[-1] if errors else "Не удалось подготовить параметры для подтверждения."
        if len(steps) >= max_steps:
            detail = "Лимит шагов исчерпан. " + detail
        answer = "Заказ не создан. " + detail
        if _has_hallucinations(answer, question, steps):
            answer = "Заказ не создан. Уточните модель, деталь, склад и количество."
        return AgentResult(answer=answer, steps=steps, sources=sources)

    if "в базе знаний не найдено" in answer.lower():
        return AgentResult(answer=REFUSAL, steps=steps, sources=sources, refused=True)

    missing_replacements = _missing_replacements(answer, question, steps)
    if _has_hallucinations(answer, question, steps) or missing_replacements:
        fix_prompt = "В ответе есть выдуманные коды, цены или остатки. Исправь ответ строго по данным инструментов."
        if missing_replacements:
            fix_prompt = (
                "Проверь коды, цены и остатки. Исправь ответ строго по данным инструментов. "
                "Ты добавил снятые артикулы без известных замен: "
                + json.dumps(missing_replacements, ensure_ascii=False)
                + ". Не рекомендуй старую деталь как актуальную. Учти всю цепочку REPLACED_BY. "
                "Если спрашивают только поставщика, назови поставщика без лишнего артикула. "
                "Старую деталь можно упомянуть вместе с её заменой."
            )
        messages.append({"role": "user", "content": fix_prompt})
        answer = llm.chat(messages).content.strip()
        if (
            _has_hallucinations(answer, question, steps)
            or _missing_replacements(answer, question, steps)
            or "в базе знаний не найдено" in answer.lower()
        ):
            return AgentResult(answer=REFUSAL, steps=steps, sources=sources, refused=True)

    return AgentResult(answer=answer, steps=steps, sources=sources, refused=False)
