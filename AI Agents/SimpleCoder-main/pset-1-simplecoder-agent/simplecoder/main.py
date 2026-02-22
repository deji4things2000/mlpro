import logging
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from simplecoder.agent import Agent


logging.basicConfig(level=logging.ERROR)
console = Console()


def main():
    """Entry point for the simple coder agent."""
    cli()


@click.command()
@click.option(
    "--model",
    default="gemini/gemini-3-flash-preview",
    help="LLM model to use"
)
@click.option(
    "--max-iterations",
    default=10,
    type=int,
    help="Maximum number of ReAct iterations"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Run in interactive mode"
)
@click.option(
    "--use-planning",
    is_flag=True,
    help="Enable planning and task decomposition"
)
@click.option(
    "--use-rag",
    is_flag=True,
    help="Enable RAG"
)
@click.option(
    "--rag-embedder",
    default="gemini/gemini-embedding-001",
    help="Embedding model for RAG"
)
@click.option(
    "--rag-index-pattern",
    default="**/*.py",
    help="File pattern for RAG"
)
@click.argument(
    "task",
    required=False
)
def cli(
    model: str,
    max_iterations: int,
    verbose: bool,
    interactive: bool,
    use_planning: bool,
    use_rag: bool,
    rag_embedder: str,
    rag_index_pattern: str,
    task: str | None
) -> None:
    """A simple coding agent."""
    agent = Agent(
        model=model,
        max_iterations=max_iterations,
        verbose=verbose,
        use_planning=use_planning,
        use_rag=use_rag,
        rag_embedder=rag_embedder,
        rag_index_pattern=rag_index_pattern
    )

    if task:
        response = agent.run(task)
        console.print(Panel(Markdown(response), title="[bold green]Agent Response[/bold green]", border_style="green"))
        return

    if interactive:
        console.print(Panel(
            "[bold cyan]SimpleCoder Agent[/bold cyan]\n\n"
            "Type your requests and I'll help you code.\n"
            "Type 'exit', 'quit', or 'q' to quit.",
            border_style="cyan"
        ))
        console.print()

        while True:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            response = agent.run(user_input)
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Agent[/bold green]",
                border_style="green"
            ))


if __name__ == "__main__":
    main()
