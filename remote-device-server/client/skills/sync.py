from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from client import api
from client.skills.base import Skill

console = Console()


class SyncSkill(Skill):
    def register(self, cli: typer.Typer) -> None:
        @cli.command()
        def upload(
            local_path: Path = typer.Argument(..., help="Local file to upload"),
        ):
            """Upload a file to the server workspace."""
            local_path = local_path.resolve()
            if not local_path.is_file():
                console.print(f"[red]Not a file: {local_path}[/red]")
                raise typer.Exit(1)
            result = api.upload_file(local_path)
            console.print(f"[green]Uploaded:[/green] {result['upload_id']} ({result['filename']})")

        @cli.command()
        def download(
            remote_path: str = typer.Argument(..., help="Remote file path"),
            local_path: Path = typer.Argument(..., help="Local destination path"),
        ):
            """Download a file from the server."""
            api.download_file(remote_path, local_path.resolve())
            console.print(f"[green]Downloaded:[/green] {local_path}")

        @cli.command()
        def envs():
            """List available conda environments on the server."""
            env_list = api.list_envs()
            if not env_list:
                console.print("[dim]No conda environments found.[/dim]")
                return
            for env in env_list:
                console.print(f"  {env['name']:20s} {env['path']}")
