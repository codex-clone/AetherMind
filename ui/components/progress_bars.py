"""
AetherMind - Rich Progress Bar Components
===========================================
Custom progress bar components for the training dashboard.
"""

from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn, TaskProgressColumn,
    MofNCompleteColumn,
)


def create_training_progress() -> Progress:
    """Create a progress bar for epoch/step tracking."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
