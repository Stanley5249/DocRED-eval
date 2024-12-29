# DocRED-eval

Evaluation of Gemini or Other LLMs on DocRED

## Installation

1. Clone the repository:
    ```shell
    git clone <repo-url>
    cd DocRED-eval
    ```

2. Set up a virtual environment and install the requirements:

    For pip, it's better to install in editable mode:
    ```shell
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -e .
    ```

    If you prefer to try `uv`, a faster alternative to pip, you can install it by:
    ```shell
    pip install uv
    uv sync
    ```

## Execution

1. Download the data from [DocRED](https://github.com/thunlp/DocRED) and place it in the `data/` folder. Only `train_annotated.json`, `test.json`, `dev.json`, and `rel_info.json` are needed.

2. Copy the `.env.example` to `.env` and fill in your API key.

3. Run the Python script:
    ```shell
    python main.py
    ```

4. Check the results in the `output/` folder. Note that sometimes Gemini may raise an error, and the new result will overwrite the previous one.

    - `messages.json`: The chat history provided by pydantic-ai.
    - `pred_labels.json`: The output from the model.
    - `true_labels.json`: The labels extracted from `data/dev.json`.
    - `system_prompt.md`: The prompt using few-shot learning from `data/train_annotated.json`.
    - `user_prompt.yaml`: The document extracted from `data/dev.json`.

## Methods

1. I use [`pydantic`](https://docs.pydantic.dev/) to validate the DocRED dataset. It has a Rust-implemented core to achieve high and reliable speed and type hints.

2. Convert the original document to a simpler version. I hypothesize that LLMs are similar to humans when understanding the document, so some entity position information and tokenized text are not important to the model. The schema for the validation is in [`docred_eval/schema.py`](docred_eval/schema.py).

3. Convert the object into YAML flow style, which might be the most token-efficient format (my intuition, not fully researched for evidence).

4. Use [`pydantic-ai`](https://ai.pydantic.dev/) to interact with the model and get the output.

## Initial Thoughts

With the rise of generative models, old ways to perform NLP tasks (customized models and complex pipelines) are no longer the most popular choice. LLMs can be used to perform a wide range of NLP tasks, but due to their **generative** nature, it is not easy to use them to extract information in a structured way.

However, I found that OpenAI provides a structured output mode for their API, which can enforce output JSON data by some unknown magic. This is different from using prompts to guide the model to generate structured output, which is way more reliable.

The problem is that I don't want to spend money on a school project, and Gemini is almost free for personal use. So I chose to evaluate Gemini on the DocRED dataset, which is a dataset for relation extraction.

I thought it would easily achieve near state-of-the-art performance, but it turns out that it is not that easy...

## The Problem

Gemini seems to refuse to output long text in structured output mode, which is a big problem for the relation extraction task.

The DocRED test set contains 3 million tokens, which is not the biggest problem for Gemini-1.5-flash, as it has a 1 million tokens context window. However, such a large input also means long output, and it has a max 8192 tokens output limit, which is not enough for the task.

<!-- Even if the limit did not exist, the model still struggles to output enough labels for the task. It often outputs 0 labels for each document, which is not acceptable. Even the Gemini-1.5-pro version with structured output mode cannot solve the problem.

But this is weird; I don't think the model is incapable of this task. Thus, I tried to use the online Gemini-1.5-flash to perform the task, and it works better than the API version even though it cannot enable structured output mode. Here is the [online case](https://g.co/gemini/share/7cbbc6513c6a).

Back to the API version, when I turn off the API version's structured output mode, it fails to output pure JSON data and generates a lot of garbage text.

So, Google might secretly limit the output length or capability of the free API version, which is not mentioned in the documentation. This is a non-responsible guess, but I have no other explanation. There is too much variance for me to figure out the real reason. -->

## Next

We failed to achieve the original goal, but I have some ideas that worked pretty well on another project using Gemini-1.5-flash structured output mode.

## Observations

`google-generative-ai` is one of the worst Python packages I've ever encountered. However, Gemini itself isn't too bad. Interestingly, the `pydantic-ai` devs feel the same way and have re-implemented the API, sharing their thoughts on [the reference page](https://ai.pydantic.dev/api/models/gemini/).

Also, the evaluation script in the `DocRED` repository is a nightmare and makes it harder to reach our goal. Nevertheless, thanks to the authors for providing the dataset.