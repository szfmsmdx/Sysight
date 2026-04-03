from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from client import api
from client.skills.base import Skill

console = Console()


class PushSkill(Skill):
    def register(self, cli: typer.Typer) -> None:
        @cli.command()
        def push(
            path: Path = typer.Argument(..., help="Local directory to push"),
            run: Optional[str] = typer.Option(None, "--run", "-r", help="Command to execute after push"),
            conda: Optional[str] = typer.Option(None, "--conda", "-c", help="Conda environment"),
        ):
            """Push local code to server and optionally run a command."""
            path = path.resolve()
            if not path.is_dir():
                console.print(f"[red]Not a directory: {path}[/red]")
                raise typer.Exit(1)

            console.print(f"[dim]Packing {path}...[/dim]")
            tar_path = _pack_directory(path)

            console.print("[dim]Uploading...[/dim]")
            result = api.upload_file(tar_path)
            upload_id = result["upload_id"]
            console.print(f"[green]Uploaded:[/green] {upload_id}")

            tar_path.unlink()

            if run:
                task = api.create_task(run, conda_env=conda, upload_id=upload_id)
                console.print(f"[green]Task submitted:[/green] {task['id']}")
                console.print(f"  Use [bold]rds logs {task['id']} -f[/bold] to follow output")


def _pack_directory(path: Path) -> Path:
    tar_path = Path(f"/tmp/rds_upload_{path.name}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(path, arcname=".")
    return tar_path
