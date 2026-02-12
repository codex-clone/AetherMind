"""
AetherMind - Training Dashboard UI
=====================================
Beautiful live terminal dashboard for monitoring training progress.
Uses Rich library for rendering.

Dependencies: rich
Used by: scripts/train.py
"""

import time
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console, Group
from typing import Dict, Any, Optional
from training.callbacks import TrainingCallback
from ui.components.metrics_panel import create_metrics_table, create_memory_table, create_speed_table
from ui.components.progress_bars import create_training_progress


class TrainingDashboard(TrainingCallback):
    """
    Rich-based live training dashboard.
    
    Displays: epoch progress, step progress, loss, metrics,
    memory usage, speed, and loss history chart.
    """
    
    def __init__(self, total_steps: int, total_epochs: int,
                  steps_per_epoch: int, model_name: str = "Forge-1-Nano"):
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        
        self.console = Console()
        self.live: Optional[Live] = None
        self.progress = create_training_progress()
        
        # State
        self.current_loss = 0.0
        self.val_loss = 0.0
        self.current_lr = 0.0
        self.tokens_per_sec = 0.0
        self.steps_per_sec = 0.0
        self.loss_history = []
        self.step_start_time = time.time()
        self.current_step = 0
        self.current_epoch = 0
    
    def _build_layout(self) -> Layout:
        """Build the dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=5),
            Layout(name="metrics", size=10),
            Layout(name="footer", size=3),
        )
        
        layout["metrics"].split_row(
            Layout(name="loss_metrics"),
            Layout(name="memory"),
            Layout(name="speed"),
        )
        
        return layout
    
    def _render(self) -> Panel:
        """Render the full dashboard."""
        import torch
        
        layout = self._build_layout()
        
        # Header
        device_info = "CPU"
        if torch.cuda.is_available():
            device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        
        header_text = Text()
        header_text.append(f"  Model: {self.model_name}", style="bold")
        header_text.append(f"  |  Device: {device_info}", style="cyan")
        header_text.append(f"  |  Phase: SFT", style="green")
        layout["header"].update(Panel(header_text, style="dim"))
        
        # Progress
        perplexity = 2.718 ** self.current_loss if self.current_loss < 20 else float("inf")
        progress_text = Text()
        progress_text.append(f"  Epoch {self.current_epoch+1}/{self.total_epochs}")
        progress_text.append(f"  |  Step {self.current_step:,}/{self.total_steps:,}")
        progress_text.append(f"  |  Loss: {self.current_loss:.4f}")
        layout["progress"].update(Panel(progress_text, title="Progress"))
        
        # Metrics panels
        layout["loss_metrics"].update(Panel(
            create_metrics_table(
                self.current_loss, self.val_loss, self.current_lr,
                perplexity, self.current_step, self.current_epoch, self.total_epochs
            ),
            title="[bold]Metrics[/bold]",
        ))
        
        layout["memory"].update(Panel(
            create_memory_table(),
            title="[bold]Memory[/bold]",
        ))
        
        eta = (self.total_steps - self.current_step) / max(self.steps_per_sec, 0.001)
        layout["speed"].update(Panel(
            create_speed_table(self.tokens_per_sec, self.steps_per_sec, eta),
            title="[bold]Speed[/bold]",
        ))
        
        # Footer
        layout["footer"].update(Panel(
            Text("  [CTRL+C to save and stop]", style="dim"),
        ))
        
        return Panel(
            layout,
            title="[bold red]\U0001f525 AetherMind Training Dashboard[/bold red]",
            border_style="red",
        )
    
    def on_train_start(self, state: Dict[str, Any]) -> None:
        self.live = Live(self._render(), console=self.console, refresh_per_second=2)
        self.live.start()
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        if self.live:
            self.live.stop()
    
    def on_step_end(self, step: int, state: Dict[str, Any]) -> None:
        self.current_step = step
        self.current_loss = state.get("loss", 0)
        self.current_lr = state.get("lr", 0)
        self.current_epoch = state.get("epoch", 0)
        self.tokens_per_sec = state.get("tokens_per_sec", 0)
        
        elapsed = time.time() - self.step_start_time
        self.steps_per_sec = step / max(elapsed, 0.001)
        
        self.loss_history.append(self.current_loss)
        
        if self.live:
            self.live.update(self._render())
