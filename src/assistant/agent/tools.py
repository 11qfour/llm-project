"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_RE = re.compile(r"^P-\d{4}[A-Z]?$")

def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
TOOL_SCHEMAS: list[dict] = [
    _tool(
        "search_knowledge",
        "Поиск по базе знаний (руководства, бюллетени, регламенты) для конкретных моделей, ошибок и деталей.",
        {"question": {"type": "string", "description": "Вопрос по модели, коду ошибки или детали"}},
        ["question"],
    ),
    _tool(
        "summarize_topic",
        "Обзорный поиск по сводкам сообществ для вопросов о статистике отказов и линейках оборудования.",
        {"question": {"type": "string", "description": "Обзорный вопрос"}},
        ["question"],
    ),
    _tool(
        "check_stock",
        "Проверить остаток детали на складе в ERP.",
        {
            "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
            "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад"},
        },
        ["part_id", "warehouse"],
    ),
    _tool(
        "get_price",
        "Узнать цену детали в рублях в ERP.",
        {"part_id": {"type": "string", "description": "Номер детали, например P-1042B"}},
        ["part_id"],
    ),
    _tool(
        "create_service_order",
        "Создать заказ на деталь в ERP.",
        {
            "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
            "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад"},
            "qty": {"type": "integer", "description": "Количество штук (> 0)"},
            "model": {"type": "string", "description": "Модель оборудования, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]

def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    def search_knowledge(question: str) -> dict:
        if not question or not isinstance(question, str):
            return {"error": "Пустой вопрос"}
        ans = local_search(question, kb, llm)
        return {"answer": ans.text, "facts": ans.facts, "sources": ans.sources, "refused": ans.refused}

    def summarize_topic(question: str) -> dict:
        if not question or not isinstance(question, str):
            return {"error": "Пустой вопрос"}
        ans = global_search(question, kb, llm)
        return {"answer": ans.text, "sources": ans.sources, "refused": ans.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        if not isinstance(part_id, str) or not PART_RE.match(part_id):
            return {"error": f"Некорректный номер детали: {part_id}"}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад: {warehouse}"}
        try:
            return erp.get_stock(part_id, warehouse)
        except ErpNotFound as exc:
            return {"error": f"Деталь не найдена в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"Запрос отклонён ERP: {exc}"}

    def get_price(part_id: str) -> dict:
        if not isinstance(part_id, str) or not PART_RE.match(part_id):
            return {"error": f"Некорректный номер детали: {part_id}"}
        try:
            return erp.get_price(part_id)
        except ErpNotFound as exc:
            return {"error": f"Деталь не найдена в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"Запрос отклонён ERP: {exc}"}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        if not isinstance(part_id, str) or not PART_RE.match(part_id):
            return {"error": f"Некорректный номер детали: {part_id}"}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад: {warehouse}"}
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": f"Количество должно быть больше нуля: {qty}"}
        if not isinstance(model, str) or not model.strip():
            return {"error": "Не указана модель"}
        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound as exc:
            return {"error": f"Деталь не найдена в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"Заказ отклонён ERP: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
