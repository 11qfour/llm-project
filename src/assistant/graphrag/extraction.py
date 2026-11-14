"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

MAX_ATTEMPTS = 2
MAX_OUTPUT_TOKENS = 4096

EntityType = Literal["ProductLine", "Model", "Component", "Part", "ErrorCode", "Procedure", "Supplier"]
RelationType = Literal[
    "BELONGS_TO", "HAS_PART", "IS_A", "INDICATES", "FIXED_BY", "REPLACED_BY", "SUPPLIED_BY", "FAILED"
]


class _EntityResponse(BaseModel):
    type: EntityType
    name: str


class _RelationResponse(BaseModel):
    subject_type: EntityType
    subject: str
    type: RelationType
    object_type: EntityType
    object: str


class _ExtractionResponse(BaseModel):
    entities: list[_EntityResponse]
    relations: list[_RelationResponse]


EXAMPLE_TEXT = "Деталь P-9001 снята с производства и заменена на P-9001B."
EXAMPLE_RESULT = {
    "entities": [{"type": "Part", "name": "P-9001"}, {"type": "Part", "name": "P-9001B"}],
    "relations": [
        {
            "subject_type": "Part",
            "subject": "P-9001",
            "type": "REPLACED_BY",
            "object_type": "Part",
            "object": "P-9001B",
        }
    ],
}


def _build_messages(chunk: Chunk, names: dict[str, list[str]]) -> list[dict]:
    """Описывает схему, направления связей и пример ответа для модели."""
    directions = "\n".join(f"{source} -{kind}-> {target}" for kind, (source, target) in RELATION_SIGNATURES.items())
    instruction = (
        "Извлеки из фрагмента сущности и связи по заданной JSON-схеме. Верни только JSON.\n"
        f"Направления разрешённых связей:\n{directions}\n"
        "REPLACED_BY направлена от старой детали к новой. FAILED означает зафиксированный отказ узла в отчёте.\n"
        "Коды бери только из фрагмента и нормализуй: ar 210 -> AR-210, Е-17 -> E-17.\n"
        f"Названия линеек, узлов и поставщиков бери из словаря: {json.dumps(names, ensure_ascii=False)}.\n"
        "Не придумывай связи, не добавляй цены, даты и номера отчётов. Перечисли сущности обоих концов связей.\n"
        "Плановое обслуживание без связей с ошибками не добавляй в граф. Если фактов по схеме нет, верни пустые списки."
    )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": EXAMPLE_TEXT},
        {"role": "assistant", "content": json.dumps(EXAMPLE_RESULT, ensure_ascii=False)},
        {"role": "user", "content": chunk.text},
    ]


def _request_extraction(messages: list[dict], llm: LLMClient) -> _ExtractionResponse | None:
    """Проверяет структуру ответа; при ошибке один раз просит модель исправить JSON."""
    for attempt in range(MAX_ATTEMPTS):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=MAX_OUTPUT_TOKENS)
        try:
            return _ExtractionResponse.model_validate_json(response.content, strict=True)
        except ValidationError as exc:
            if attempt + 1 < MAX_ATTEMPTS:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": f"Исправь формат ответа: {exc}. Верни JSON по схеме."})
    return None


def _normalize_entity(kind: str, name: str, names: dict[str, list[str]], codes: set[str]) -> Entity | None:
    """Проверяет код по тексту чанка, название по словарю и возвращает каноническую сущность."""
    canonical = canonical_name(kind, name, names)
    if canonical is None or (kind in CODE_TYPES and canonical not in codes):
        return None
    return Entity(type=kind, name=canonical)


def _to_result(data: _ExtractionResponse, chunk: Chunk, names: dict[str, list[str]]) -> ExtractionResult:
    """Нормализует сущности и связи; каждый отброшенный элемент увеличивает dropped."""
    result = ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
    codes = set(find_codes(chunk.text))
    entities: dict[str, Entity] = {}
    for item in data.entities:
        entity = _normalize_entity(item.type, item.name, names, codes)
        if entity is None:
            result.dropped += 1
        else:
            entities[entity.node_id] = entity

    for item in data.relations:
        subject = _normalize_entity(item.subject_type, item.subject, names, codes)
        obj = _normalize_entity(item.object_type, item.object, names, codes)
        if subject is None or obj is None:
            result.dropped += 1
            continue
        relation = Relation(subject=subject, type=item.type, object=obj)
        if not relation.matches_schema():
            result.dropped += 1
            continue
        entities[subject.node_id] = subject
        entities[obj.node_id] = obj
        if relation not in result.relations:
            result.relations.append(relation)
    result.entities = list(entities.values())
    return result


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Извлекает сущности и связи по FR-1; после двух некорректных ответов возвращает пустой результат."""
    names = vocabulary()
    data = _request_extraction(_build_messages(chunk, names), llm)
    if data is None:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
    return _to_result(data, chunk, names)
