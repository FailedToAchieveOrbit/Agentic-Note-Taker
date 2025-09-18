"""Modern async CLI for the robust note-taking system.

This module provides a user-friendly command-line interface using Rich for
beautiful output and async operations for better performance.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .config import Settings
from .database import AsyncNoteDatabase

app = typer.Typer(
    name="rnote",
    help="🗒️  Robust Note Taker - AI-powered semantic search for your notes",
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print("Robust Note Taker v2.0.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            help="Show version and exit",
        ),
    ] = None,
    config_path: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Initialize the application."""
    if config_path:
        # Custom configuration loading would go here
        pass


@app.command("add")
def add_note(
    content: Annotated[
        Optional[str],
        typer.Option(
            "--content",
            "-c",
            help="Note content (will prompt if not provided)",
        ),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option(
            "--file",
            "-f",
            help="Read content from file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    tags: Annotated[
        Optional[list[str]],
        typer.Option(
            "--tag",
            "-t",
            help="Add tags (can be used multiple times)",
        ),
    ] = None,
) -> None:
    """📝 Add a new note to your collection."""
    asyncio.run(_add_note_async(content, file, tags or []))


async def _add_note_async(
    content: Optional[str], file: Optional[Path], tags: list[str]
) -> None:
    """Async implementation of adding a note."""
    try:
        settings = Settings()
        async with AsyncNoteDatabase(settings.data_dir) as db:
            # Get content from various sources
            if file:
                content = await asyncio.to_thread(file.read_text, encoding="utf-8")
            elif not content:
                content = typer.prompt("Enter your note")

            if not content.strip():
                console.print("❌ [red]Error:[/red] Empty note content")
                raise typer.Exit(1)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Adding note and generating embedding...")
                
                note_id = await db.add_note(content, tags=tags)
                progress.update(task, completed=True)

            console.print(f"✅ [green]Note added successfully![/green] ID: [cyan]{note_id}[/cyan]")
            
            if tags:
                tags_text = ", ".join(f"[blue]#{tag}[/blue]" for tag in tags)
                console.print(f"🏷️  Tags: {tags_text}")

    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ [red]Error adding note:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_notes(
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            help="Maximum number of notes to show",
            min=1,
        ),
    ] = 20,
    tag: Annotated[
        Optional[str],
        typer.Option(
            "--tag",
            "-t",
            help="Filter by tag",
        ),
    ] = None,
    show_content: Annotated[
        bool,
        typer.Option(
            "--content",
            help="Show full content instead of preview",
        ),
    ] = False,
) -> None:
    """📋 List all notes in your collection."""
    asyncio.run(_list_notes_async(limit, tag, show_content))


async def _list_notes_async(
    limit: int, tag: Optional[str], show_content: bool
) -> None:
    """Async implementation of listing notes."""
    try:
        settings = Settings()
        async with AsyncNoteDatabase(settings.data_dir) as db:
            notes = await db.list_notes(limit=limit, tag_filter=tag)

            if not notes:
                if tag:
                    console.print(f"📭 No notes found with tag [blue]#{tag}[/blue]")
                else:
                    console.print("📭 No notes found. Use [cyan]rnote add[/cyan] to create your first note!")
                return

            # Create a beautiful table
            table = Table(title=f"📚 Your Notes {f'(#{tag})' if tag else ''}")
            table.add_column("ID", style="cyan", no_wrap=True, width=8)
            table.add_column("Content", style="white")
            table.add_column("Tags", style="blue", no_wrap=True)
            table.add_column("Created", style="dim", no_wrap=True)

            for note in notes:
                note_id = str(note["id"])[:8]
                content = note["content"]
                if not show_content and len(content) > 100:
                    content = content[:97] + "..."
                
                tags_str = ", ".join(f"#{tag}" for tag in note.get("tags", []))
                created = note.get("created_at", "Unknown")[:19]
                
                table.add_row(note_id, content, tags_str, created)

            console.print(table)
            console.print(f"\n📊 Total: [cyan]{len(notes)}[/cyan] notes")

    except Exception as e:
        console.print(f"❌ [red]Error listing notes:[/red] {e}")
        raise typer.Exit(1)


@app.command("search")
def search_notes(
    query: Annotated[str, typer.Argument(help="Search query")],
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of top results to return",
            min=1,
            max=50,
        ),
    ] = 5,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help="Search method: semantic, bm25, or hybrid",
            case_sensitive=False,
        ),
    ] = "hybrid",
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Minimum similarity threshold (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.1,
    llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Generate LLM-powered answer from search results",
        ),
    ] = True,
) -> None:
    """🔍 Search through your notes using AI-powered semantic search."""
    asyncio.run(_search_notes_async(query, top_k, method.lower(), threshold, llm))


async def _search_notes_async(
    query: str, top_k: int, method: str, threshold: float, use_llm: bool
) -> None:
    """Async implementation of searching notes."""
    if method not in ["semantic", "bm25", "hybrid"]:
        console.print(f"❌ [red]Error:[/red] Invalid method '{method}'. Use: semantic, bm25, or hybrid")
        raise typer.Exit(1)

    try:
        settings = Settings()
        async with AsyncNoteDatabase(settings.data_dir) as db:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"Searching using {method} method...")
                
                results = await db.search(
                    query=query,
                    top_k=top_k,
                    method=method,
                    threshold=threshold,
                )
                progress.update(task, completed=True)

            if not results:
                console.print(f"🔍 No results found for: [yellow]'{query}'[/yellow]")
                console.print("💡 Try:
  • Different keywords
  • Lower threshold value
  • Different search method")
                return

            # Display search results
            console.print(f"\n🔍 [bold]Search Results for:[/bold] [yellow]'{query}'[/yellow]")
            console.print(f"📊 Method: [cyan]{method}[/cyan] | Found: [green]{len(results)}[/green] results\n")

            for i, (note, score) in enumerate(results, 1):
                score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
                console.print(f"[bold cyan]{i}. Score: [{score_color}]{score:.3f}[/{score_color}][/bold cyan]")
                
                # Show preview
                content = note["content"]
                if len(content) > 200:
                    content = content[:197] + "..."
                console.print(f"   {content}")
                
                # Show tags and metadata
                if note.get("tags"):
                    tags_str = ", ".join(f"[blue]#{tag}[/blue]" for tag in note["tags"])
                    console.print(f"   🏷️  {tags_str}")
                
                console.print(f"   [dim]ID: {str(note['id'])[:8]} | Created: {note.get('created_at', 'Unknown')[:19]}[/dim]\n")

            # Generate LLM answer if requested and available
            if use_llm and settings.openai_api_key:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("Generating AI-powered answer...")
                    
                    answer = await db.generate_answer(query, results)
                    progress.update(task, completed=True)

                if answer:
                    console.print("\n🤖 [bold blue]AI-Generated Answer:[/bold blue]")
                    console.print(f"[italic]{answer}[/italic]")
            elif use_llm and not settings.openai_api_key:
                console.print("\n⚠️  [yellow]AI answer unavailable: OpenAI API key not configured[/yellow]")

    except KeyboardInterrupt:
        console.print("\n⚠️  Search cancelled")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ [red]Error during search:[/red] {e}")
        raise typer.Exit(1)


@app.command("stats")
def show_stats() -> None:
    """📊 Show statistics about your note collection."""
    asyncio.run(_show_stats_async())


async def _show_stats_async() -> None:
    """Async implementation of showing stats."""
    try:
        settings = Settings()
        async with AsyncNoteDatabase(settings.data_dir) as db:
            stats = await db.get_stats()
            
            console.print("\n📊 [bold blue]Note Collection Statistics[/bold blue]\n")
            
            table = Table(show_header=False)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")
            
            table.add_row("📝 Total Notes", str(stats.get("total_notes", 0)))
            table.add_row("🏷️  Total Tags", str(stats.get("total_tags", 0)))
            table.add_row("📏 Average Length", f"{stats.get('avg_length', 0):.0f} characters")
            table.add_row("📅 Oldest Note", stats.get("oldest_note", "N/A"))
            table.add_row("🆕 Newest Note", stats.get("newest_note", "N/A"))
            table.add_row("💾 Database Size", stats.get("db_size", "N/A"))
            
            console.print(table)
            
            # Show most common tags
            if stats.get("top_tags"):
                console.print("\n🏷️  [bold]Most Common Tags:[/bold]")
                for tag, count in stats["top_tags"][:5]:
                    console.print(f"   [blue]#{tag}[/blue]: {count} notes")

    except Exception as e:
        console.print(f"❌ [red]Error getting stats:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()