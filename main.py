"""
Command‑line interface for the note‑taking system.

This script allows you to add notes, list stored notes and query them
using natural language.  It relies only on the Python standard
library and scikit‑learn, making it suitable for offline environments.
"""

from __future__ import annotations

import argparse
import os
from typing import List

from note_taker.database import NoteDatabase


def handle_add(db: NoteDatabase, args: argparse.Namespace) -> None:
    """Add a note to the database."""
    db.add_note(args.content)
    print("Note added successfully.")


def handle_list(db: NoteDatabase, args: argparse.Namespace) -> None:
    """List all notes stored in the database."""
    if not db.notes:
        print("No notes found. Use the 'add' command to store a note.")
        return
    for idx, note in enumerate(db.notes, start=1):
        preview = note['content'][:80]
        suffix = '…' if len(note['content']) > 80 else ''
        print(f"{idx}. {note['id']} – {preview}{suffix}")


def handle_ask(db: NoteDatabase, args: argparse.Namespace) -> None:
    """Handle a query against the stored notes and optionally invoke a language model."""
    if not db.notes:
        print("No notes have been added yet. Use the 'add' command first.")
        return
    results = db.search(args.question, top_k=args.top_k)
    if not results:
        print("No notes found.")
        return
    print("Top relevant notes:")
    context_blocks: List[str] = []
    for rank, (note, score) in enumerate(results, start=1):
        preview = note['content'][:100]
        suffix = '…' if len(note['content']) > 100 else ''
        print(f"{rank}. (score {score:.4f}) {preview}{suffix}")
        context_blocks.append(f"Note {rank}: {note['content']}")
    # Attempt to call OpenAI for a synthesised answer
    try:
        import openai  # type: ignore
        # Ensure API key is available
        if not os.environ.get('OPENAI_API_KEY'):
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        system_prompt = (
            "You are a helpful assistant answering questions based on provided notes. "
            "Use only the information contained in the notes to answer the question. "
            "If the notes do not contain relevant information, reply that you don't know."
        )
        context = "\n\n".join(context_blocks)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Notes:\n{context}\n\nQuestion: {args.question}\n\nAnswer concisely.",
            },
        ]
        response = openai.ChatCompletion.create(model=args.model, messages=messages)
        answer = response.choices[0].message['content'].strip()
        print("\nAnswer:\n" + answer)
    except Exception:
        # Silence any exceptions related to OpenAI; retrieval results have already been shown.
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage and query notes using retrieval‑augmented generation techniques."
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new note to the database')
    add_parser.add_argument('--content', '-c', required=True, help='The text content of the note')
    add_parser.set_defaults(func=handle_add)

    # List command
    list_parser = subparsers.add_parser('list', help='List all stored notes')
    list_parser.set_defaults(func=handle_list)

    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Query your notes with a question')
    ask_parser.add_argument('question', help='The question to ask')
    ask_parser.add_argument('--top-k', type=int, default=3, help='Number of top notes to retrieve')
    ask_parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model for answer synthesis')
    ask_parser.set_defaults(func=handle_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    db = NoteDatabase()
    # Dispatch to the appropriate handler
    args.func(db, args)


if __name__ == '__main__':
    main()