import typer
from rich.console import Console

from client.skills.submit import SubmitSkill
from client.skills.logs import LogsSkill
from client.skills.push import PushSkill
from client.skills.monitor import MonitorSkill
from client.skills.sync import SyncSkill

app = typer.Typer(
    name="rds",
    help="Remote Device Server CLI — manage tasks on GPU machines.",
    no_args_is_help=True,
)
console = Console()

# Register all skills
for skill_cls in [SubmitSkill, LogsSkill, PushSkill, MonitorSkill, SyncSkill]:
    skill_cls().register(app)


@app.command()
def health():
    """Check server connectivity."""
    from client import api

    try:
        result = api.health()
        console.print(f"[green]Server OK[/green]: {result}")
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
