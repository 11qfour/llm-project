"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def _add_entity(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    """Объединяет одинаковые сущности и накапливает уникальные чанки-источники."""
    if entity.node_id not in graph:
        graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=set())
    graph.nodes[entity.node_id]["sources"].add(chunk_id)


def _resolve_replacements(graph: nx.DiGraph) -> None:
    """Убирает более слабую встречную замену, а при равных весах обе."""
    for source, target in list(graph.edges):
        if source == target or not graph.has_edge(source, target) or not graph.has_edge(target, source):
            continue
        forward = graph.edges[source, target]
        backward = graph.edges[target, source]
        if forward["type"] != "REPLACED_BY" or backward["type"] != "REPLACED_BY":
            continue

        graph.graph["conflicts"] += 1
        if forward["weight"] >= backward["weight"]:
            graph.remove_edge(target, source)
        if forward["weight"] <= backward["weight"]:
            graph.remove_edge(source, target)


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        for entity in result.entities:
            _add_entity(graph, entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            _add_entity(graph, relation.subject, result.chunk_id)
            _add_entity(graph, relation.object, result.chunk_id)
            source, target = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, sources=set())
            graph.edges[source, target]["sources"].add(result.chunk_id)

    for _, node in graph.nodes(data=True):
        node["sources"] = sorted(node["sources"])
    for _, _, edge in graph.edges(data=True):
        edge["sources"] = sorted(edge["sources"])
        edge["weight"] = len(edge["sources"])

    _resolve_replacements(graph)
    return graph
