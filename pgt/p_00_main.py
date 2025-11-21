from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold magenta]Hello pgraph-tutorial[/bold magenta]",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    main()
