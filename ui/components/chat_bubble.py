"""
AetherMind - Chat Bubble Components
======================================
Styled message components for the terminal chat interface.
"""

from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown


def user_message(text: str) -> Panel:
    """Create a styled user message bubble."""
    return Panel(
        Text(text, style="white"),
        title="[bold cyan]You[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )


def assistant_message(text: str, thinking: str = "",
                       show_thinking: bool = False,
                       gen_time: float = 0, num_tokens: int = 0) -> Panel:
    """Create a styled assistant message bubble with optional thinking."""
    content_parts = []
    
    if thinking and show_thinking:
        content_parts.append(Text("\U0001f4ad Thinking:", style="dim italic"))
        content_parts.append(Text(thinking, style="dim"))
        content_parts.append(Text(""))
    elif thinking:
        content_parts.append(Text("\U0001f4ad [Thinking...] (press T to expand)", style="dim italic"))
        content_parts.append(Text(""))
    
    content_parts.append(Markdown(text))
    
    # Build subtitle with stats
    subtitle = ""
    if gen_time > 0:
        subtitle = f"{gen_time:.1f}s | {num_tokens} tok"
    
    from rich.console import Group
    return Panel(
        Group(*content_parts),
        title="[bold green]Forge-1[/bold green]",
        subtitle=f"[dim]{subtitle}[/dim]" if subtitle else None,
        border_style="green",
        padding=(0, 1),
    )


def system_message(text: str) -> Panel:
    """Create a system/info message."""
    return Panel(
        Text(text, style="yellow"),
        border_style="yellow dim",
        padding=(0, 1),
    )
