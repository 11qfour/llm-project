"""Контракт FR-6 – FR-10: инструменты и цикл агента."""

import pytest

from assistant.agent.agent import _numbers_without_codes, run_agent
from assistant.agent.tools import TOOL_SCHEMAS, build_tools
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM

NAMES = ["search_knowledge", "summarize_topic", "check_stock", "get_price", "create_service_order"]
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
ORDER = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}


def call(name: str, **arguments) -> LLMResponse:
    return LLMResponse("", [ToolCall(f"call-{name}", name, arguments)])


class Tools:
    """Подставные инструменты: считают вызовы и отдают заранее заданные результаты."""

    def __init__(self, **overrides) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._results = {
            "search_knowledge": {
                "answer": "E-17 – отказ пускового реле. В AR-210 это P-1042, она заменена на P-1042B.",
                "facts": ["Part:P-1042 -REPLACED_BY-> Part:P-1042B"],
                "sources": ["bulletin_SB-2025-03.md"],
                "refused": False,
            },
            "summarize_topic": {"answer": "Чаще всего отказывает пусковое реле.", "sources": [], "refused": False},
            "check_stock": {"part_id": "P-1042B", "warehouse": "Казань", "qty": 3, "discontinued": False},
            "get_price": {"part_id": "P-1042B", "name": "пусковое реле", "price_rub": 2630, "discontinued": False},
            "create_service_order": {"order_id": "SO-000001", "status": "created", **ORDER},
            **overrides,
        }

    def as_dict(self) -> dict:
        def make(name: str):
            def tool(**arguments) -> dict:
                self.calls.append((name, arguments))
                result = self._results[name]
                return result(**arguments) if callable(result) else result

            return tool

        return {name: make(name) for name in NAMES}

    def count(self, name: str) -> int:
        return sum(1 for called, _ in self.calls if called == name)


def test_tool_schemas_describe_the_five_tools():
    by_name = {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}
    assert sorted(by_name) == sorted(NAMES)
    assert all(schema["type"] == "function" for schema in TOOL_SCHEMAS)
    stock = by_name["check_stock"]["function"]["parameters"]
    assert stock["properties"]["warehouse"]["enum"] == WAREHOUSES and sorted(stock["required"]) == [
        "part_id",
        "warehouse",
    ]
    order = by_name["create_service_order"]["function"]["parameters"]
    assert sorted(order["required"]) == ["model", "part_id", "qty", "warehouse"]
    assert by_name["search_knowledge"]["function"]["parameters"]["required"] == ["question"]


def test_build_tools_returns_five_functions(kb, erp):
    assert sorted(build_tools(kb, FakeLLM([]), erp)) == sorted(NAMES)


def test_erp_tools_return_values_from_the_erp(kb, erp):
    tools = build_tools(kb, FakeLLM([]), erp)
    stock = tools["check_stock"](part_id="P-1042B", warehouse="Казань")
    assert stock["qty"] == 3 and stock["discontinued"] is False and "error" not in stock
    assert tools["get_price"](part_id="P-1042B")["price_rub"] == 2630
    assert tools["create_service_order"](**ORDER)["order_id"] == "SO-000001"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("check_stock", {"part_id": "P-1042B", "warehouse": "Екатеринбург"}),
        ("check_stock", {"part_id": "реле", "warehouse": "Казань"}),
        ("get_price", {"part_id": "AR-210"}),
        ("create_service_order", {**ORDER, "qty": 0}),
        ("create_service_order", {**ORDER, "warehouse": "Склад №5"}),
    ],
)
def test_bad_arguments_are_rejected_before_the_erp_is_called(kb, erp, name, arguments):
    result = build_tools(kb, FakeLLM([]), erp)[name](**arguments)
    assert "error" in result and erp.calls == []


def test_missing_part_and_zero_stock_are_different_results(kb, erp):
    tools = build_tools(kb, FakeLLM([]), erp)
    zero = tools["check_stock"](part_id="P-1042B", warehouse="Москва")
    missing = tools["check_stock"](part_id="P-9999", warehouse="Москва")
    assert zero["qty"] == 0 and "error" not in zero
    assert "error" in missing and "qty" not in missing


def test_rejected_order_is_reported_as_an_error(kb, erp):
    result = build_tools(kb, FakeLLM([]), erp)["create_service_order"](**{**ORDER, "part_id": "P-1042"})
    assert "error" in result and "снята с производства" in result["error"]


def test_search_knowledge_returns_answer_facts_and_sources(kb, erp):
    llm = FakeLLM(["P-1042 заменена на P-1042B."])
    result = build_tools(kb, llm, erp)["search_knowledge"](question="Чем заменена деталь P-1042?")
    assert result["answer"] == "P-1042 заменена на P-1042B." and result["refused"] is False
    assert "bulletin_SB-2025-03.md" in result["sources"] and any("P-1042B" in fact for fact in result["facts"])


def test_agent_chains_knowledge_search_and_erp(kb):
    tools = Tools()
    llm = FakeLLM(
        [
            call("search_knowledge", question="Какую деталь заказать при ошибке E-17 на AR-210?"),
            call("get_price", part_id="P-1042B"),
            "Закажите P-1042B, цена 2630 руб.",
        ]
    )
    result = run_agent("Ошибка E-17 на AR-210. Что заказать и сколько стоит?", llm, tools.as_dict())
    assert result.answer == "Закажите P-1042B, цена 2630 руб." and not result.refused
    assert [(step.tool, step.error) for step in result.steps] == [("search_knowledge", None), ("get_price", None)]
    assert result.steps[1].args == {"part_id": "P-1042B"} and result.steps[1].result["price_rub"] == 2630
    assert "bulletin_SB-2025-03.md" in result.sources
    assert llm.requests[0]["tools"]  # модель видит описания инструментов


def test_part_number_that_appeared_nowhere_is_not_sent_to_the_tool():
    tools = Tools()
    llm = FakeLLM([call("get_price", part_id="P-1043"), "Номер детали мне неизвестен."])
    result = run_agent("Сколько стоит пусковое реле для AR-210?", llm, tools.as_dict())
    assert tools.count("get_price") == 0
    assert result.steps[0].tool == "get_price" and result.steps[0].error
    assert result.answer == "Номер детали мне неизвестен."


def test_unknown_tool_and_broken_arguments_come_back_to_the_model_as_errors():
    tools = Tools()
    llm = FakeLLM([call("delete_everything"), call("get_price", _raw="мусор"), "Не получилось."])
    result = run_agent("Сколько стоит P-1042B?", llm, tools.as_dict())
    assert tools.calls == [] and len(result.steps) == 2 and all(step.error for step in result.steps)
    assert result.answer == "Не получилось."


def test_identical_call_is_executed_once():
    tools = Tools()
    llm = FakeLLM([call("get_price", part_id="P-1042B"), call("get_price", part_id="P-1042B"), "Цена 2630 руб."])
    result = run_agent("Сколько стоит P-1042B?", llm, tools.as_dict())
    assert tools.count("get_price") == 1 and result.answer == "Цена 2630 руб."


def test_agent_stops_after_max_steps_tool_calls():
    tools = Tools()
    script = [call("check_stock", part_id="P-1042B", warehouse=warehouse) for warehouse in WAREHOUSES]
    llm = FakeLLM([*script, "Не успел закончить: проверены не все склады."])
    result = run_agent("Где есть P-1042B?", llm, tools.as_dict(), max_steps=2)
    assert tools.count("check_stock") == 2 and result.answer


def test_unavailable_erp_is_retried_twice():
    attempts = []

    def flaky(**arguments) -> dict:
        attempts.append(1)
        if len(attempts) < 3:
            raise ErpUnavailable("ERP недоступна")
        return {"part_id": "P-1042B", "warehouse": "Казань", "qty": 3, "discontinued": False}

    tools = Tools(check_stock=flaky)
    llm = FakeLLM([call("check_stock", part_id="P-1042B", warehouse="Казань"), "На складе в Казани 3 шт."])
    result = run_agent("Есть ли P-1042B в Казани?", llm, tools.as_dict())
    assert len(attempts) == 3 and result.steps[0].error is None and result.steps[0].result["qty"] == 3
    assert result.answer == "На складе в Казани 3 шт."


def test_erp_that_never_answers_is_reported_and_nothing_is_invented():
    attempts = []

    def dead(**arguments) -> dict:
        attempts.append(1)
        raise ErpUnavailable("ERP недоступна")

    tools = Tools(check_stock=dead)
    llm = FakeLLM([call("check_stock", part_id="P-1042B", warehouse="Казань"), "Данные ERP получить не удалось."])
    result = run_agent("Есть ли P-1042B в Казани?", llm, tools.as_dict())
    assert len(attempts) == 3 and result.steps[0].error
    assert result.answer == "Данные ERP получить не удалось."


def test_order_waits_for_confirmation():
    tools = Tools()
    llm = FakeLLM([call("create_service_order", **ORDER), "Жду подтверждения."])
    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.", llm, tools.as_dict())
    assert tools.count("create_service_order") == 0
    assert result.needs_confirmation and result.pending_action == {"tool": "create_service_order", "args": ORDER}


def test_confirmed_order_is_created():
    tools = Tools()
    llm = FakeLLM([call("create_service_order", **ORDER), "Заказ SO-000001 создан."])
    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.", llm, tools.as_dict(), confirm=True)
    assert tools.calls == [("create_service_order", ORDER)]
    assert not result.needs_confirmation and result.pending_action is None and "SO-000001" in result.answer


def test_order_without_a_quantity_named_by_the_user_is_not_created_even_when_confirmed():
    tools = Tools()
    guessed = {**ORDER, "qty": 1}
    llm = FakeLLM([call("create_service_order", **guessed), "Сколько штук заказать?"])
    result = run_agent("Закажи P-1042B для AR-210 на склад Казань.", llm, tools.as_dict(), confirm=True)
    assert tools.count("create_service_order") == 0
    assert not result.needs_confirmation and result.pending_action is None and result.answer


def test_answer_with_an_invented_price_is_sent_back_for_correction():
    tools = Tools()
    llm = FakeLLM([call("get_price", part_id="P-1042B"), "Цена 4850 руб.", "Цена 2630 руб."])
    result = run_agent("Сколько стоит P-1042B?", llm, tools.as_dict())
    assert len(llm.requests) == 3 and result.answer == "Цена 2630 руб." and not result.refused


def test_answer_that_stays_invented_becomes_a_refusal():
    tools = Tools()
    llm = FakeLLM(
        [call("get_price", part_id="P-1042B"), "Закажите P-7777 за 4850 руб.", "Закажите P-7777 за 4850 руб."]
    )
    result = run_agent("Сколько стоит P-1042B?", llm, tools.as_dict())
    assert result.refused and "4850" not in result.answer and "P-7777" not in result.answer


def test_refusal_of_the_knowledge_base_is_passed_on():
    refused = {"answer": "В базе знаний не найдено.", "facts": [], "sources": [], "refused": True}
    tools = Tools(search_knowledge=refused)
    llm = FakeLLM([call("search_knowledge", question="Что означает ошибка E-99?"), "В базе знаний не найдено."])
    result = run_agent("Что означает ошибка E-99 на AR-210?", llm, tools.as_dict())
    assert result.refused and result.answer == "В базе знаний не найдено."

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Закажи 12 штук", {"12"}),
        ("SO-123456", set()),
        ("AR-210", set()),
        ("АR-210", set()),  # русская А + латинская R
        ("Оформи 2 шт. P-1042B для AR-210 при E-17 по SP-104", {"2"}),
        ("Бюллетень SB-2025-03 и отчёт R-2025-017: 5 выездов", {"5"}),
        ("Деталь р-1042в для ar 210 при Е-17 по цене 2630 руб.", {"2630"}),
    ],
)
def test_numbers_without_codes_handles_edge_cases(text, expected):
    assert _numbers_without_codes(text) == expected


def test_repeated_cached_calls_still_respect_max_steps():
    tools = Tools()
    repeated = [call("get_price", part_id="P-1042B") for _ in range(5)]
    llm = FakeLLM([*repeated, "Прервано по лимиту шагов."])
    result = run_agent("Сколько стоит P-1042B?", llm, tools.as_dict(), max_steps=2)
    assert tools.count("get_price") == 1 and len(result.steps) == 2


def test_extra_unexpected_argument_is_rejected_without_crash():
    tools = Tools()
    llm = FakeLLM([call("get_price", part_id="P-1042B", warehouse="Казань"), "Цена неизвестна."])
    result = run_agent("Сколько стоит P-1042B?", llm, tools.as_dict())
    assert tools.count("get_price") == 0 and result.steps[0].error


def test_part_digits_are_not_accepted_as_invented_price():
    tools = Tools()
    llm = FakeLLM([call("get_price", part_id="P-1042B"), "Цена 1042 руб.", "Цена 2630 руб."])
    result = run_agent("Сколько стоит P-1042B?", llm, tools.as_dict())
    assert len(llm.requests) == 3 and result.answer == "Цена 2630 руб."


def test_invalid_order_args_are_not_put_into_pending_action():
    tools = Tools()
    bad_order = {**ORDER, "warehouse": "Екатеринбург"}
    llm = FakeLLM([call("create_service_order", **bad_order), "Такого склада нет."])
    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Екатеринбург.", llm, tools.as_dict())
    assert not result.needs_confirmation and result.pending_action is None and result.steps[0].error


def test_hallucinated_confirmation_message_is_checked():
    tools = Tools()
    llm = FakeLLM(
        [
            call("create_service_order", **ORDER),
            "Подтвердите заказ P-7777 за 9999 руб.",
            "Подтвердите заказ 2 шт. P-1042B для AR-210.",
        ]
    )
    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.", llm, tools.as_dict())
    assert result.needs_confirmation and "P-7777" not in result.answer and "9999" not in result.answer