from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.gemini import GeminiModel
from pydantic_core import to_json

from docred_eval.project import GEMINI_API_KEY, dir_data, dir_output
from docred_eval.schema import *
from docred_eval.utils import save_text, yaml_dump_compact

# ==============================================================================
# Make the system prompt
# ==============================================================================

# use the first 3 examples for few shot learning
simple_docred_train = load_and_validate_simple_docred(
    dir_data / "train_annotated.json",
    s=slice(10),
)

simple_docred_example = SimpleDocRED.model_validate(
    [
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
    ]
)

system_prompt = f"""\
## Document Relation Extraction Task  

You are a machine learning engineer working on a document relation extraction task. The dataset is a simplified version of the DocRED dataset in YAML format. Your task is to predict a list of documents in JSON format. Be thorough and confident in identifying relations and their evidence.  

### Guidelines for Predictions  

1. **Output Format**  
   Return a JSON array of documents. Each document must include its title and a list of extracted labels. Do not add any explanations or extra text.  

2. **Comprehensive Predictions**  
   Extract as many relations as possible, including all relevant evidence. Multiple relations and evidence can exist for a single entity pair.  

3. **Evaluation Metric**  
   Your predictions will be evaluated using the micro F1 score, which emphasizes both accuracy and coverage of extracted relations.  

### Schema  

Each document in the output should have:  

- `title`: The title of the document.  
- `labels`: A list of relationships between entities, where each label includes:  
  - `r`: The relation code (e.g., "P17" for "country").  
  - `h`: The head entity ID.  
  - `t`: The tail entity ID.  
  - `evidence`: A list of sentence IDs supporting the relation.  

Additional fields in the training data:  

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
{yaml_dump_compact(simple_docred_example.model_dump_features())}\
```

### Example Output (JSON)
```json
{simple_docred_example.model_dump_labels_json()}
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

save_text(dir_output / "system_prompt.md", system_prompt)

# ==============================================================================
# Make the user prompt
# ==============================================================================

# use the first 3 test data for user prompt
simple_docred_test = load_and_validate_simple_docred(
    dir_data / "dev.json",
    s=slice(1),
)

user_prompt = yaml_dump_compact(simple_docred_test.model_dump_features())

save_text(dir_output / "user_prompt.yaml", user_prompt)
save_text(
    dir_output / "true_labels.json", simple_docred_test.model_dump_labels_json(indent=2)
)

# ==============================================================================
# Run the model
# ==============================================================================

geimini = GeminiModel(
    "gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
)

agent = Agent(
    model=geimini,
    result_type=list[ResultDocument],
    system_prompt=system_prompt,
    model_settings={"temperature": 0.0},
)

try:
    result = agent.run_sync(user_prompt=user_prompt)

except UnexpectedModelBehavior:
    raise

else:
    print(result.usage())

    save_text(
        dir_output / "messages.json",
        to_json(result.all_messages(), indent=2).decode(),
    )

    save_text(
        dir_output / "pred_labels.json",
        to_json(result.data, indent=2).decode(),
    )
