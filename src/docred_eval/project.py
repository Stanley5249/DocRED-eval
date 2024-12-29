from os import PathLike, environ
from pathlib import Path

__all__ = [
    "StrPath",
    "GEMINI_API_KEY",
    "dir_data",
    "dir_output",
]


type StrPath = str | PathLike[str]

GEMINI_API_KEY = environ["GEMINI_API_KEY"]

dir_data = Path("data")

dir_output = Path("output")
