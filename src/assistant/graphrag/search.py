"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from assistant.graphrag.communities import _summary_context
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, Community, KnowledgeBase

MAX_PATH_LENGTH = 3
NEIGHBOURHOOD_DEPTH = 2
MAX_FACT_CHARS = 8000
MAX_ANSWER_TOKENS = 1024
MAX_MAP_TOKENS = 512
REFUSAL = "В базе знаний не найдено."


class _MapAnswer(BaseModel):
    answer: str
    score: int = Field(ge=0, le=10)


def _replacement_edges(graph: nx.DiGraph, parts: set[str]) -> set[tuple[str, str]]:
    """Собирает направленные цепочки замен; visited защищает от циклов в графе."""
    edges: set[tuple[str, str]] = set()
    visited: set[str] = set()
    pending = list(parts)
    while pending:
        part = pending.pop()
        if part in visited:
            continue
        visited.add(part)
        for _, target, attrs in graph.out_edges(part, data=True):
            if attrs["type"] == "REPLACED_BY":
                edges.add((part, target))
                pending.append(target)
    return edges


def _local_edges(graph: nx.DiGraph, anchors: list[str]) -> set[tuple[str, str]]:
    """Ищет пути без учёта направления, но сохраняет исходные направления фактов."""
    undirected = graph.to_undirected(as_view=True)
    if len(anchors) == 1:
        nodes = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=NEIGHBOURHOOD_DEPTH)
        return set(graph.subgraph(nodes).edges)

    edges: set[tuple[str, str]] = set()
    for source, target in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, source, target, cutoff=MAX_PATH_LENGTH):
            for left, right in pairwise(path):
                for edge in ((left, right), (right, left)):
                    if graph.has_edge(*edge):
                        edges.add(edge)
    parts = {node for edge in edges for node in edge if graph.nodes[node]["type"] == "Part"}
    return edges | _replacement_edges(graph, parts)


def _part_details(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Добавляет тип и поставщика только уже найденных деталей, включая конечные замены."""
    parts = {node for edge in edges for node in edge if graph.nodes[node]["type"] == "Part"}
    return {
        (part, target)
        for part in parts
        for _, target, attrs in graph.out_edges(part, data=True)
        if attrs["type"] in {"IS_A", "SUPPLIED_BY"}
    }


def _local_context(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Укладывает целые факты в лимит; источники берёт только из вошедших рёбер."""
    facts: list[str] = []
    sources: set[str] = set()
    length = 0
    # При ограничении контекста сначала сохраняем сведения о замене старых деталей.
    ordered = sorted(edges, key=lambda edge: (graph.edges[edge]["type"] != "REPLACED_BY", edge))
    for source, target in ordered:
        attrs = graph.edges[source, target]
        fact = f"{graph.nodes[source]['name']} -{attrs['type']}-> {graph.nodes[target]['name']}"
        line = f"{len(facts) + 1}. {fact}"
        added_length = len(line) + bool(facts)
        if length + added_length > MAX_FACT_CHARS:
            continue
        facts.append(fact)
        length += added_length
        sources.update(chunk_id.split("#", 1)[0] for chunk_id in attrs["sources"])
    return facts, sorted(sources)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    anchors = {f"{code_type(code)}:{code}" for code in find_codes(question)}
    if not anchors.issubset(kb.graph.nodes):
        return Answer(text=REFUSAL, refused=True)
    anchors.update(
        f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary) if f"{kind}:{name}" in kb.graph
    )
    if not anchors:
        return Answer(text=REFUSAL, refused=True)

    edges = _local_edges(kb.graph, sorted(anchors))
    edges |= _part_details(kb.graph, edges)
    facts, sources = _local_context(kb.graph, edges)
    if not facts:
        return Answer(text=REFUSAL, refused=True)

    context = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    messages = [
        {
            "role": "system",
            "content": (
                "Ты ассистент сервисной службы. Ответь кратко по-русски только по перечисленным фактам графа. "
                "HAS_PART связывает модель с деталью, IS_A — деталь с видом узла, "
                "INDICATES — ошибку с узлом, FIXED_BY — ошибку с регламентом. "
                "SUPPLIED_BY связывает деталь с поставщиком, BELONGS_TO — модель с линейкой, "
                "FAILED — модель с отказавшим узлом. "
                "REPLACED_BY направлена от старой детали к новой: пройди всю показанную цепочку "
                "и при выборе детали для нового заказа назови последнюю деталь. "
                "Если спрашивают, в каких моделях стояла старая деталь, ответь по HAS_PART для этой детали. "
                "Если спрашивают поставщика, используй SUPPLIED_BY актуальной детали. "
                "Если вопрос просит только модели или регламент, не заменяй ответ рекомендацией детали. "
                "Не добавляй коды, цены и остатки, которых нет в фактах. "
                f"Если фактов для ответа недостаточно, ответь ровно: {REFUSAL}"
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\n\nФакты графа:\n{context}"},
    ]
    text = llm.chat(messages, max_tokens=MAX_ANSWER_TOKENS).content.strip()
    refused = not text or "в базе знаний не найдено" in text.lower()
    return Answer(text=text or REFUSAL, sources=sources, facts=facts, refused=refused)


def _map_summary(question: str, summary: str, llm: LLMClient) -> _MapAnswer | None:
    """Получает частичный ответ по одной сводке; неверный JSON или бесполезный ответ пропускает."""
    messages = [
        {
            "role": "system",
            "content": (
                "Ответь на вопрос только по данной сводке сообщества. Верни JSON по схеме: "
                "answer — краткий частичный ответ, score — целая оценка его полезности от 0 до 10. "
                "Если сводка не содержит сведений для ответа, верни answer пустой строкой и score=0. "
                "Отвечай именно о запрошенной линейке: сведения о другой линейке имеют score=0. "
                "В answer укажи линейку вместе с результатом. Для вопросов о частоте используй готовые "
                "строки 'Отказы узлов' или 'Отказы моделей'; не считай разные виды узлов числом отказов. "
                "Строка 'Снятые детали' явно называет старые детали и их замены. "
                "Сохрани относящиеся к вопросу модели, линейки, узлы и готовые числа из сводки. "
                "Статистика одной группы не описывает весь корпус. Не добавляй факты из самого вопроса "
                "или собственных знаний."
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\n\nСводка:\n{summary}"},
    ]
    response = llm.chat(messages, json_schema=MAP_SCHEMA, max_tokens=MAX_MAP_TOKENS)
    try:
        partial = _MapAnswer.model_validate_json(response.content, strict=True)
    except ValidationError:
        return None
    return partial if partial.score > 0 and partial.answer.strip() else None


def _entities_supported(text: str, context: str, names: dict[str, list[str]]) -> bool:
    """Проверяет, что коды и названия из переданного словаря встречаются в исходном контексте."""
    return set(find_codes(text)).issubset(find_codes(context)) and set(find_names(text, names)).issubset(
        find_names(context, names)
    )


def _community_sources(community: Community, graph: nx.DiGraph) -> set[str]:
    """Источники тех же связей, которые использовались для сводки, включая граничные."""
    return set(_summary_context(community, graph)["sources"])


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    if any(f"{code_type(code)}:{code}" not in kb.graph for code in find_codes(question)):
        return Answer(text=REFUSAL, refused=True)

    partials: list[dict] = []
    sources: set[str] = set()
    line_names = {"ProductLine": kb.vocabulary.get("ProductLine", [])}
    asked_lines = set(find_names(question, line_names))
    for community in kb.communities:
        partial = _map_summary(question, community.summary, llm)
        summary_lines = set(find_names(community.summary, line_names))
        if asked_lines and summary_lines and asked_lines.isdisjoint(summary_lines):
            continue
        # Оценка модели не подтверждает факт: нельзя приписывать сводке другую линейку.
        if partial is not None and _entities_supported(partial.answer, community.summary, line_names):
            partials.append(
                {
                    "community_id": community.community_id,
                    "answer": partial.answer.strip(),
                    "summary": community.summary,
                }
            )
            sources.update(_community_sources(community, kb.graph))
    if not partials:
        return Answer(text=REFUSAL, refused=True)

    messages = [
        {
            "role": "system",
            "content": (
                "Объедини частичные ответы разных сообществ в один краткий ответ по-русски. "
                "Сверяй частичные ответы с приложенными исходными сводками, убери повторы. "
                "Короткий частичный ответ может не повторять линейку: она указана в его сводке. "
                "Готовые строки 'Отказы узлов', 'Отказы моделей' и 'Снятые детали' сохраняются кодом. "
                "Используй их для сравнения и перечисления, не подменяй числа количеством разных видов узлов. "
                "Если искомые данные явно есть в сводках, дай ответ. Учитывай, к каким моделям "
                "и линейкам относится каждый вывод. Сопоставь сведения всех полезных сообществ, "
                "не выдавай вывод одной группы за результат по всему корпусу. "
                "Не придумывай коды, числа, цены и остатки. Не упоминай номера сообществ. "
                f"Если сведений недостаточно или они противоречат друг другу, ответь ровно: {REFUSAL}"
            ),
        },
        {
            "role": "user",
            "content": f"Вопрос: {question}\n\nЧастичные ответы:\n{json.dumps(partials, ensure_ascii=False)}",
        },
    ]
    text = llm.chat(messages, max_tokens=MAX_ANSWER_TOKENS).content.strip()
    context = question + "\n" + "\n".join(partial["summary"] + "\n" + partial["answer"] for partial in partials)
    if not _entities_supported(text, context, kb.vocabulary):
        text = REFUSAL
    refused = not text or "в базе знаний не найдено" in text.lower()
    return Answer(text=text or REFUSAL, sources=sorted(sources), refused=refused)
