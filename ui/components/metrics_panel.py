"""
AetherMind - Metrics Panel Component
=======================================
Live-updating metrics display for the training dashboard.
"""

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import torch


def create_metrics_table(loss: float, val_loss: float, lr: float,
                          perplexity: float, step: int, epoch: int,
                          total_epochs: int) -> Table:
    """Create a metrics table for the dashboard."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Train Loss", f"{loss:.4f}")
    table.add_row("Val Loss", f"{val_loss:.4f}" if val_loss else "N/A")
    table.add_row("Perplexity", f"{perplexity:.2f}")
    table.add_row("Learning Rate", f"{lr:.2e}")
    table.add_row("Step", f"{step:,}")
    table.add_row("Epoch", f"{epoch+1}/{total_epochs}")
    
    return table


def create_memory_table() -> Table:
    """Create a memory usage table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        util = (allocated / total) * 100 if total > 0 else 0
        
        table.add_row("VRAM Used", f"{allocated:.1f} GB / {total:.1f} GB")
        table.add_row("VRAM Reserved", f"{reserved:.1f} GB")
        table.add_row("GPU Util", f"{util:.0f}%")
    else:
        table.add_row("Device", "CPU")
    
    return table


def create_speed_table(tokens_per_sec: float, steps_per_sec: float,
                        eta_seconds: float) -> Table:
    """Create a speed metrics table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Tokens/sec", f"{tokens_per_sec:,.0f}")
    table.add_row("Steps/sec", f"{steps_per_sec:.2f}")
    
    if eta_seconds > 0:
        hours = int(eta_seconds // 3600)
        mins = int((eta_seconds % 3600) // 60)
        table.add_row("ETA", f"{hours}h {mins}m")
    
    return table
