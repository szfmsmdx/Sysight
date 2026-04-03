from __future__ import annotations

import json
import time

import typer
from rich.console import Console

from client import api
from client.config import API_KEY, SERVER_URL
from client.skills.base import Skill

console = Console()


class LogsSkill(Skill):
    def register(self, cli: typer.Typer) -> None:
        @cli.command()
        def logs(
            task_id: str = typer.Argument(..., help="Task ID"),
            follow: bool = typer.Option(False, "--follow", "-f", help="Stream logs in real time"),
        ):
            """View task logs."""
            if follow:
                _follow_ws(task_id)
            else:
                data = api.get_logs(task_id)
                if data["data"]:
                    console.print(data["data"], end="")
                else:
                    console.print("[dim]No logs yet.[/dim]")


def _follow_ws(task_id: str) -> None:
    try:
        from websockets.sync.client import connect
    except ImportError:
        console.print("[red]websockets package required for --follow[/red]")
        raise typer.Exit(1)

    ws_url = SERVER_URL.replace("http://", "ws://").replace("https://", "wss://")
    url = f"{ws_url}/tasks/{task_id}/logs/ws"
    headers = {"X-API-Key": API_KEY}

    try:
        with connect(url, additional_headers=headers) as ws:
            for message in ws:
                print(message, end="", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]WebSocket error: {e}[/red]")
