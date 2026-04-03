from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from client import api
from client.skills.base import Skill

console = Console()


class MonitorSkill(Skill):
    def register(self, cli: typer.Typer) -> None:
        @cli.command()
        def monitor(
            watch: bool = typer.Option(False, "--watch", "-w", help="Continuously refresh"),
            interval: float = typer.Option(2.0, "--interval", "-i", help="Refresh interval in seconds"),
        ):
            """Show system resource usage (CPU, memory, GPU, disk)."""
            if watch:
                _watch_loop(interval)
            else:
                data = api.get_monitor()
                _print_snapshot(data)


def _build_table(data: dict) -> Table:
    table = Table(title="System Monitor", show_header=False, expand=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    # CPU
    cpus = data.get("cpu_percent", [])
    if cpus:
        avg = sum(cpus) / len(cpus)
        table.add_row("CPU", f"{avg:.1f}% avg ({len(cpus)} cores)")

    # Memory
    mem = data.get("memory", {})
    if mem:
        table.add_row("Memory", f"{mem['used_gb']:.1f} / {mem['total_gb']:.1f} GB ({mem['percent']}%)")

    # Disk
    disk = data.get("disk", {})
    if disk:
        table.add_row("Disk", f"{disk['used_gb']:.1f} / {disk['total_gb']:.1f} GB ({disk['percent']}%)")

    # GPUs
    for gpu in data.get("gpus", []):
        table.add_row(
            f"GPU {gpu['index']}",
            f"{gpu['name']}  util={gpu['utilization_percent']}%  "
            f"mem={gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB  "
            f"temp={gpu['temperature_c']}°C",
        )

    return table


def _print_snapshot(data: dict) -> None:
    console.print(_build_table(data))


def _watch_loop(interval: float) -> None:
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                data = api.get_monitor()
                live.update(_build_table(data))
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
