from collections.abc import Iterable, Mapping
from math import inf
from pathlib import Path
from typing import Any

import yaml

from docred_eval.project import StrPath


def invert_dict[K, V](d: Mapping[K, V]) -> dict[V, K]:
    return {v: k for k, v in d.items()}


def to_unique_list(li: Iterable[Any]) -> list[Any]:
    return [*dict.fromkeys(li)]


def yaml_dump_compact(obj: Any) -> str:
    return yaml.safe_dump(obj, default_flow_style=True, width=inf, sort_keys=False)


def save_text(path: StrPath, text: str) -> int:
    n_char = Path(path).write_text(text)
    print(f"Saved {n_char} characters to {path}")
    return n_char
