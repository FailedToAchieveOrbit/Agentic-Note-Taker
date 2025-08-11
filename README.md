# Robust Note Taker for Retrieval‑Augmented Generation (RAG)

This repository implements a simple yet powerful note‑taking system that
stores your notes locally and allows you to query them using natural language.  It
uses TF‑IDF vectorisation and cosine similarity to measure the relevance of
notes to a query.  When an OpenAI API key is available, the tool can
optionally call ChatGPT to synthesise an answer from the retrieved notes.  When
no key or OpenAI library is present the tool simply returns the most
relevant notes.

## Features

* **Add notes** – Save free‑form text into a local database.  Each note
  is stored persistently on disk.
* **Query your knowledge base** – Ask questions in plain language and
  receive the most relevant notes.  Cosine similarity is used to
  measure relevance.
* **LLM‑powered answers** – When an OpenAI API key is provided via
  `OPENAI_API_KEY` and the `openai` Python package is available, the
  tool can generate a concise, synthesized answer from the retrieved
  notes using ChatGPT (GPT‑3.5 or GPT‑4).
* **Offline fallback** – If no OpenAI key or library is set, the system
  still functions entirely offline using scikit‑learn’s TF‑IDF vectoriser
  and returns the top notes without generating an answer.

## Requirements

* Python ≥ 3.9
* [Optional] An OpenAI API key for embedding and answer generation

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you intend to use the offline fallback only, you may omit the
`openai` package from the installation by removing it from
`requirements.txt`.

## Usage

To see available commands run:

```bash
python main.py -h
```

### Adding a note

Add a note by providing its content.  The note will be stored and
embedded automatically.

```bash
python main.py add --content "Marie Curie was a pioneer in the study of radioactivity."
```

### Listing notes

Display all stored notes and their IDs:

```bash
python main.py list
```

### Asking a question

Query your knowledge base using natural language.  By default the top
three relevant notes are returned.  If an OpenAI API key is set the
tool will also call ChatGPT to generate a concise answer.

```bash
python main.py ask "Who pioneered the study of radioactivity?" --top-k 3
```

The retrieved notes will be printed along with similarity scores.
When `OPENAI_API_KEY` is set in your environment, an additional
language‑model‑generated answer will follow.

## Data storage

All notes are saved to `data/notes.json` within this repository.
Removing this file resets the database.

## License

This project is licensed under the MIT License; see the `LICENSE`
file for details.