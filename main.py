import asyncio
import json
from collections import deque
from functools import partial
from typing import Annotated, Any, TypedDict

from aiolimiter import AsyncLimiter
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.gemini import GeminiModel
from pydantic_core import to_json

from docred_eval.project import GEMINI_API_KEY, dir_data, dir_output
from docred_eval.schema import *
from docred_eval.schema import ListModel
from docred_eval.utils import save_text, yaml_dump_compact

type DocEvalTask = tuple[int, SimpleDocument]


class Relation(TypedDict):
    title: str
    r: RelationEnum
    h: Annotated[int, Field(serialization_alias="h_idx")]
    t: Annotated[int, Field(serialization_alias="t_idx")]
    evidence: list[int]


class Submission(ListModel[Relation]):
    pass


def make_system_prompt() -> str:
    simple_docred_train = load_and_validate_simple_docred(
        dir_data / "train_annotated.json",
        s=slice(50),
    )

    simple_doc_example = SimpleDocument.model_validate(
        {
            "title": "Example Document",
            "labels": [
                {"r": "P131", "h": 0, "t": 1, "evidence": [0]},
                {"r": "P17", "h": 1, "t": 2, "evidence": [1]},
            ],
            "sents": [
                {"id": 0, "text": "Entity A is located in City B."},
                {"id": 1, "text": "City B is part of Country C."},
            ],
            "entities": [
                {"id": 0, "type": "Entity", "names": ["Entity A"], "sent_ids": [0]},
                {"id": 1, "type": "City", "names": ["City B"], "sent_ids": [0, 1]},
                {"id": 2, "type": "Country", "names": ["Country C"], "sent_ids": [1]},
            ],
        }
    )

    return f"""\
## Document Relation Extraction Task  

You are a machine learning engineer working on a document relation extraction task. The dataset is a simplified version of the DocRED dataset in YAML format. Your task is to predict a list of labels in JSON format. Be thorough and confident in identifying relations and their evidence.  

### Guidelines for Predictions  

1. **Output Format**  
Return a JSON array of labels. Do not add any explanations or extra text.  

2. **Comprehensive Predictions**  
Extract as many relations as possible, including all relevant evidence. Multiple relations and evidence can exist for a single entity pair.  

3. **Evaluation Metric**  
Your predictions will be evaluated using the micro F1 score, which emphasizes both accuracy and coverage of extracted relations.  

### Schema  

Each label in the output should have:  

- `r`: The relation code (e.g., "P17" for "country").  
- `h`: The head entity ID.  
- `t`: The tail entity ID.  
- `evidence`: A list of sentence IDs supporting the relation.  

Each document in training data have:

- `title`
- `labels`: Same as the output labels.
- `sents`: A list of sentences in the document:  
- `id`: Sentence ID.  
- `text`: The content of the sentence.  
- `entities`: A list of entities in the document:  
- `id`: Entity ID.  
- `type`: The type of the entity.
- `names`: Names or aliases of the entity.  
- `sent_ids`: The sentence IDs where the entity appears.

### Example Input (YAML)
```yaml
{yaml_dump_compact(simple_doc_example.model_dump_features())}\
```

### Example Output (JSON)
```json
{to_json(simple_doc_example.labels, by_alias=True).decode()}
```

### Relation Information
```yaml
{yaml_dump_compact(REL_INFO_DICT)}\
```

### Training Data
```yaml
{yaml_dump_compact(simple_docred_train.model_dump(mode="json", by_alias=True))}\
```\
"""


def callback[T: tuple[int, Any]](
    task: asyncio.Task[Any],
    *,
    done: list[T],
    queue: deque[T],
    item: T,
) -> None:
    if task.cancelled():
        queue.appendleft(item)
    else:
        done.append(item)

    print(f"done {item[0]}")


async def run_agent[T](
    agent: Agent[None, T],
    i: int,
    doc: SimpleDocument,
) -> T:
    print(f"send {i}")

    obj = doc.model_dump_features()
    prompt = yaml_dump_compact(obj)

    try:
        result = await agent.run(prompt)

    except Exception:
        raise asyncio.CancelledError()

    else:
        return result.data


async def eval_docred(
    *,
    docred: SimpleDocRED,
    model: Model,
    system_prompt: str,
    limiter: AsyncLimiter,
) -> Submission:
    agent = Agent(
        model=model,
        result_type=list[Label],
        system_prompt=system_prompt,
        model_settings={"temperature": 0.0},
        retries=0,
    )

    n_docs = len(docred.root)

    done: list[DocEvalTask] = []

    queue: deque[DocEvalTask] = deque(enumerate(docred.root))

    tasks: dict[int, asyncio.Task[list[Label]]] = {}

    async with asyncio.TaskGroup() as g:
        while len(done) < n_docs:
            async with limiter:
                if not queue:
                    continue

                i, doc = item = queue.popleft()
                task = g.create_task(run_agent(agent, i, doc))
                task.add_done_callback(
                    partial(callback, done=done, queue=queue, item=item)
                )
                tasks[i] = task

    relations = [
        {"title": doc.title, **label}
        for i, doc in enumerate(docred.root)
        for label in tasks[i].result()
    ]

    return Submission.model_validate(relations)


def main() -> None:
    docred = load_and_validate_simple_docred(dir_data / "test.json")

    model = GeminiModel("gemini-1.5-flash", api_key=GEMINI_API_KEY)

    system_prompt = make_system_prompt()
    save_text(dir_output / "system_prompt.md", system_prompt)

    limiter = AsyncLimiter(14, 60)

    task = eval_docred(
        docred=docred,
        model=model,
        system_prompt=system_prompt,
        limiter=limiter,
    )

    submission = asyncio.run(task)

    save_text(
        dir_output / "result.json",
        json.dumps(submission.model_dump(mode="json", by_alias=True)),
    )


if __name__ == "__main__":
    main()
