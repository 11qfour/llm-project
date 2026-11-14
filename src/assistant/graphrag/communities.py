"""Сообщества графа и их сводки. Требование FR-3."""

import json
from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

LOUVAIN_SEED = 7
MAX_SUMMARY_WORDS = 150
MAX_SUMMARY_TOKENS = 1024


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for source, target, attrs in sorted(graph.edges(data=True)):
        weight = attrs["weight"]
        if undirected.has_edge(source, target):
            undirected.edges[source, target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    if undirected.number_of_edges() == 0:
        groups = [{node} for node in undirected.nodes]
    else:
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=LOUVAIN_SEED)
    return [
        Community(community_id=number, nodes=nodes)
        for number, nodes in enumerate(sorted(sorted(group) for group in groups))
    ]


def _summary_context(community: Community, graph: nx.DiGraph) -> dict:
    """Собирает факты внутри группы и считает частоты отказов по весам FAILED."""
    subgraph = graph.subgraph(community.nodes)
    model_lines: dict[str, list[str]] = {}
    # Линейка может оказаться в другой группе, но нужна для описания её моделей.
    for model, line, attrs in sorted(graph.out_edges(subgraph.nodes, data=True)):
        if attrs["type"] == "BELONGS_TO":
            model_lines.setdefault(model, []).append(line)

    relations = []
    failures: Counter[str] = Counter()
    failures_by_line: Counter[tuple[str, str]] = Counter()
    for source, target, attrs in sorted(subgraph.edges(data=True)):
        relations.append({"source": source, "type": attrs["type"], "target": target, "weight": attrs["weight"]})
        if attrs["type"] == "FAILED":
            failures[target] += attrs["weight"]
            for line in model_lines.get(source, []):
                failures_by_line[line, target] += attrs["weight"]

    return {
        "nodes": sorted(subgraph.nodes),
        "relations": relations,
        "model_lines": model_lines,
        "total_failure_mentions": sum(failures.values()),
        "failure_mentions_by_component": dict(sorted(failures.items(), key=lambda item: (-item[1], item[0]))),
        "failure_mentions_by_line": [
            {"line": line, "component": component, "count": count}
            for (line, component), count in sorted(failures_by_line.items())
        ],
    }


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    context = _summary_context(community, graph)
    messages = [
        {
            "role": "system",
            "content": (
                f"Напиши по-русски сводку сообщества графа, не более {MAX_SUMMARY_WORDS} слов. "
                "Назови модели и линейки, связи деталей и наиболее частые отказы. "
                "Используй только переданные факты. Числа уже посчитаны кодом: не пересчитывай их. "
                "Вес FAILED означает число чанков с упоминанием данного отказа. "
                "Сумма этих весов не означает число уникальных выездов. "
                "Веса остальных связей не считай отказами. "
                "model_lines даёт принадлежность моделей к линейкам, включая линейки вне сообщества. "
                "Статистика относится только к связям внутри этого сообщества, а не ко всему корпусу."
            ),
        },
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    response = llm.chat(messages, max_tokens=MAX_SUMMARY_TOKENS)
    return " ".join(response.content.split()[:MAX_SUMMARY_WORDS])
