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
    """Учитывает граничные связи; каждый FAILED принадлежит группе своей модели."""
    owned = set(community.nodes)
    edges = set(graph.out_edges(owned))
    for node in owned:
        if graph.nodes[node]["type"] == "Part":
            edges.update(
                (model, node) for model, _, attrs in graph.in_edges(node, data=True) if attrs["type"] == "HAS_PART"
            )
    models = {node for edge in edges for node in edge if graph.nodes[node]["type"] == "Model"}
    model_lines: dict[str, list[str]] = {}
    for model in sorted(models):
        for _, line, attrs in graph.out_edges(model, data=True):
            if attrs["type"] == "BELONGS_TO":
                edges.add((model, line))
                model_lines.setdefault(model, []).append(line)

    relations = []
    sources: set[str] = set()
    failures: Counter[str] = Counter()
    failures_by_line: Counter[tuple[str, str]] = Counter()
    failures_by_model: Counter[tuple[str, str]] = Counter()
    replacements: set[tuple[str, str, str, str]] = set()
    for source, target in sorted(edges):
        attrs = graph.edges[source, target]
        relations.append({"source": source, "type": attrs["type"], "target": target, "weight": attrs["weight"]})
        sources.update(chunk.split("#", 1)[0] for chunk in attrs["sources"])
        if attrs["type"] == "FAILED" and source in owned:
            failures[target] += attrs["weight"]
            for line in model_lines.get(source, []):
                failures_by_line[line, target] += attrs["weight"]
                failures_by_model[line, source] += attrs["weight"]
        if attrs["type"] == "REPLACED_BY" and source in owned:
            for model, _, incoming in graph.in_edges(source, data=True):
                if incoming["type"] == "HAS_PART":
                    for line in model_lines.get(model, []):
                        replacements.add((line, model, source, target))
    if not edges:
        sources.update(chunk.split("#", 1)[0] for node in owned for chunk in graph.nodes[node]["sources"])
    return {
        "nodes": sorted(owned),
        "relations": relations,
        "model_lines": model_lines,
        "total_failure_mentions": sum(failures.values()),
        "failure_mentions_by_component": dict(failures),
        "failure_mentions_by_line": [
            {"line": line, "component": component, "count": count}
            for (line, component), count in sorted(failures_by_line.items())
        ],
        "failure_mentions_by_model": [
            {"line": line, "model": model, "count": count} for (line, model), count in sorted(failures_by_model.items())
        ],
        "replacements": [
            {"line": line, "model": model, "old": old, "new": new} for line, model, old, new in sorted(replacements)
        ],
        "sources": sorted(sources),
    }


def _summary_facts(context: dict) -> list[str]:
    """Сохраняет точные подсчёты и замены в сводке, даже если LLM их пропустит."""
    lines = []
    for key, label, entity in (
        ("failure_mentions_by_line", "Отказы узлов", "component"),
        ("failure_mentions_by_model", "Отказы моделей", "model"),
    ):
        grouped: dict[str, list[str]] = {}
        for row in context[key]:
            line = row["line"].split(":", 1)[1]
            name = row[entity].split(":", 1)[1]
            grouped.setdefault(line, []).append(f"{name}: {row['count']}")
        for line, values in grouped.items():
            lines.append(f"{label}, линейка «{line}»: " + "; ".join(values) + ".")
    replacements: dict[str, set[str]] = {}
    for row in context["replacements"]:
        line = row["line"].split(":", 1)[1]
        old, new = row["old"].split(":", 1)[1], row["new"].split(":", 1)[1]
        replacements.setdefault(line, set()).add(f"{old} → {new}")
    for line, pairs in replacements.items():
        lines.append(f"Снятые детали моделей линейки «{line}» и их замены: " + "; ".join(sorted(pairs)) + ".")
    return lines


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Одно обращение к модели; сводка и сохранённые кодом факты занимают до 150 слов."""
    context = _summary_context(community, graph)
    facts = []
    used = 0
    # Целые факты идут первыми: обрезать числовую строку или цепочку замены нельзя.
    for line in _summary_facts(context):
        size = len(line.split())
        if used + size <= MAX_SUMMARY_WORDS:
            facts.append(line)
            used += size
    messages = [
        {
            "role": "system",
            "content": (
                "Кратко опиши по-русски модели, линейки, связи деталей и поставщиков в сообществе. "
                f"Не более {MAX_SUMMARY_WORDS - used} слов. Используй только переданные связи. "
                "Не считай и не сравнивай отказы: код добавит готовые подсчёты и сведения о заменах отдельно. "
                "Статистика FAILED охватывает все отказы моделей данной группы, включая связи с узлами вне группы. "
                "Числа обозначают упоминания в чанках, а не количество уникальных выездов. "
                "Упоминание двух деталей в одной группе не означает, что они взаимозаменяемы."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({k: v for k, v in context.items() if k != "sources"}, ensure_ascii=False),
        },
    ]
    response = llm.chat(messages, max_tokens=MAX_SUMMARY_TOKENS)
    prose = " ".join(response.content.split()[: MAX_SUMMARY_WORDS - used])
    return "\n".join([*facts, prose]).strip()
