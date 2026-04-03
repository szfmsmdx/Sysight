from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from client import api
from client.skills.base import Skill

console = Console()


class SubmitSkill(Skill):
    def register(self, cli: typer.Typer) -> None:
        @cli.command()
        def run(
            command: str = typer.Argument(..., help="Shell command to execute"),
            conda: Optional[str] = typer.Option(None, "--conda", "-c", help="Conda environment name"),
            workdir: Optional[str] = typer.Option(None, "--workdir", "-w", help="Working directory on server"),
        ):
            """Submit a task for remote execution."""
            task = api.create_task(command, conda_env=conda, working_dir=workdir)
            console.print(f"[green]Task submitted:[/green] {task['id']}")
            console.print(f"  status: {task['status']}")
            console.print(f"  command: {task['command']}")

        @cli.command()
        def cancel(task_id: str = typer.Argument(..., help="Task ID to cancel")):
            """Cancel a running task."""
            result = api.cancel_task(task_id)
            console.print(f"[yellow]{result['status']}[/yellow]")

        @cli.command(name="ps")
        def list_tasks(
            limit: int = typer.Option(20, "--limit", "-n"),
        ):
            """List recent tasks."""
            data = api.list_tasks(limit=limit)
            for t in data["tasks"]:
                status_color = {
                    "running": "blue",
                    "success": "green",
                    "failed": "red",
                    "pending": "yellow",
                    "cancelled": "dim",
                }.get(t["status"], "white")
                console.print(
                    f"[{status_color}]{t['status']:>10}[/{status_color}]  "
                    f"{t['id']}  {t['command'][:60]}"
                )
            console.print(f"\n[dim]Total: {data['total']}[/dim]")

        @cli.command()
        def info(task_id: str = typer.Argument(..., help="Task ID")):
            """Show task details."""
            t = api.get_task(task_id)
            for k, v in t.items():
                if v is not None:
                    console.print(f"  [bold]{k}:[/bold] {v}")
