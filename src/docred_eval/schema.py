from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Self, TypedDict

from pydantic import BaseModel, Field, RootModel, TypeAdapter

from docred_eval.project import StrPath, dir_data
from docred_eval.utils import invert_dict, to_unique_list

__all__ = [
    "REL_INFO_DICT",
    "RelationEnum",
    "Label",
    "Mention",
    "Entity",
    "Sentence",
    "Document",
    "SimpleDocument",
    "DocRED",
    "SimpleDocRED",
    "load_and_validate_simple_docred",
    "ResultDocument",
    "ResultDocRED",
]

# ==============================================================================
# enum
# ==============================================================================

REL_INFO_ADAPTER = TypeAdapter(dict[str, str])

REL_INFO_DICT = REL_INFO_ADAPTER.validate_json(
    Path(dir_data / "rel_info.json").read_bytes()
)

RelationEnum = Enum("RelationEnum", invert_dict(REL_INFO_DICT))


# ==============================================================================
# data models (TypedDict)
# ==============================================================================


class Label(TypedDict):
    """Represents a labeled relation between two entities, supported by evidence sentences."""

    r: Annotated[
        RelationEnum,
        Field(description="Relation code corresponding to the `rel_info` mapping."),
    ]
    h: Annotated[
        int,
        Field(description="ID of the head entity, matching an `entities.id` field."),
    ]
    t: Annotated[
        int,
        Field(description="ID of the tail entity, matching an `entities.id` field."),
    ]
    evidence: Annotated[
        list[int],
        Field(
            description="List of sentence IDs from `sents.id` that support the relation.",
        ),
    ]


class Mention(TypedDict):
    """Represents a mention of an entity in a document. Used to build the item in `Document.vertex_set`."""

    name: str
    pos: tuple[int, int]
    sent_id: int


class Entity(TypedDict):
    """Represents a named entity in a document. Used to build `SimpleDocument.entities`."""

    id: int
    names: list[str]
    sent_ids: list[int]


class Sentence(TypedDict):
    """Represents a sentence in a document. Used to build `SimpleDocument.sents`."""

    id: int
    text: str


# ==============================================================================
# document models
# ==============================================================================


class BaseDocument(BaseModel):
    title: str
    labels: list[Label] | None = None


class Document(BaseDocument):
    model_config = {"populate_by_name": True}

    sents: list[list[str]]
    vertex_set: Annotated[list[list[Mention]], Field(alias="vertexSet")]


class SimpleDocument(BaseDocument):
    sents: list[Sentence]
    entities: list[Entity]

    @classmethod
    def model_validate_doc(cls, doc: Document) -> Self:
        sentence: list[Sentence] = [
            {
                "id": i,
                "text": " ".join(sent),
            }
            for i, sent in enumerate(doc.sents)
        ]

        vertex_set: list[Entity] = [
            {
                "id": i,
                "names": to_unique_list([item["name"] for item in mentions]),
                "sent_ids": to_unique_list([item["sent_id"] for item in mentions]),
            }
            for i, mentions in enumerate(doc.vertex_set)
        ]
        return cls(
            title=doc.title,
            labels=doc.labels,
            sents=sentence,
            entities=vertex_set,
        )


# ==============================================================================
# top-level model
# ==============================================================================


class ListModel[T](RootModel[list[T]]):
    root: Annotated[list[T], Field(fail_fast=True)]

    def apply_slice(self, s: slice | None) -> Self:
        update = None if s is None else {"root": self.root[s]}
        return self.model_copy(update=update)


class DocRED(ListModel[Document]):
    pass


class SimpleDocRED(ListModel[SimpleDocument]):
    @classmethod
    def model_validate_docred(cls, docred: DocRED) -> Self:
        return cls([*map(SimpleDocument.model_validate_doc, docred.root)])

    def model_dump_features(self, **kwargs: Any) -> Any:
        return self.model_dump(
            mode="json",
            by_alias=True,
            exclude={"__all__": {"labels"}},
            **kwargs,
        )

    def model_dump_features_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(
            by_alias=True,
            exclude={"__all__": {"labels"}},
            **kwargs,
        )

    def model_dump_labels(self, **kwargs: Any) -> Any:
        return self.model_dump(
            mode="json",
            by_alias=True,
            include={"__all__": {"title", "labels"}},
            **kwargs,
        )

    def model_dump_labels_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(
            by_alias=True,
            include={"__all__": {"title", "labels"}},
            **kwargs,
        )


def load_and_validate_simple_docred(
    path: StrPath, *, s: slice | None = None
) -> SimpleDocRED:
    data = Path(path).read_bytes()
    docred = DocRED.model_validate_json(data).apply_slice(s)
    return SimpleDocRED.model_validate_docred(docred)


# ==============================================================================
# genai resulst type
# ==============================================================================


class ResultDocument(BaseModel):
    """Represents the relation extraction results for a single document."""

    title: Annotated[str, Field(description="The title of the document.")]
    labels: Annotated[
        list[Label],
        Field(
            description="List of labeled relations (`labels`) extracted for the document."
        ),
    ]


class ResultDocRED(ListModel[ResultDocument]):
    pass
