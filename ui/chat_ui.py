"""
AetherMind - Terminal Chat Interface
=======================================
Interactive chat UI using Rich for beautiful terminal output.

Features:
  - Styled user/assistant messages
  - Collapsible thinking blocks (toggle with /think)
  - Token streaming
  - Generation stats
  - Slash commands (/help, /quit, /clear, /save, /think, /model)

Dependencies: rich, inference/thinking_pipeline.py
Used by: scripts/chat.py
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from ui.components.chat_bubble import user_message, assistant_message, system_message

logger = logging.getLogger(__name__)


class ChatInterface:
    """
    Terminal chat interface for Forge-1.
    
    Args:
        model: Loaded Forge1Model
        tokenizer: Forge1Tokenizer
        device: Compute device
        show_thinking: Whether to show thinking blocks by default
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda",
                  show_thinking: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.show_thinking = show_thinking
        self.console = Console()
        self.conversation_history: List[Dict] = []
    
    def display_banner(self):
        """Show the welcome banner."""
        banner = Text()
        banner.append("\n")
        banner.append("    \U0001f525 FORGE-1  ", style="bold red")
        banner.append("|  AetherMind SLM  ", style="bold")
        banner.append("|  v0.1.0\n", style="dim")
        banner.append("    Type /help for commands\n", style="dim italic")
        
        self.console.print(Panel(
            banner,
            border_style="bright_red",
            padding=(0, 2),
        ))
    
    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if should continue, False to exit."""
        cmd = cmd.strip().lower()
        
        if cmd in ("/quit", "/exit"):
            self.console.print("\n[dim]Goodbye! \U0001f44b[/dim]")
            return False
        
        elif cmd == "/clear":
            self.conversation_history.clear()
            self.console.clear()
            self.display_banner()
            self.console.print(system_message("Conversation cleared."))
        
        elif cmd == "/think":
            self.show_thinking = not self.show_thinking
            status = "ON" if self.show_thinking else "OFF"
            self.console.print(system_message(f"Thinking display: {status}"))
        
        elif cmd == "/save":
            self._save_conversation()
        
        elif cmd == "/help":
            help_text = (
                "/quit or /exit - Exit the chat\n"
                "/clear - Clear conversation history\n"
                "/think - Toggle thinking/reasoning display\n"
                "/save - Save conversation to file\n"
                "/model - Show model info\n"
                "/help - Show this help"
            )
            self.console.print(system_message(help_text))
        
        elif cmd == "/model":
            n_params = self.model.count_parameters()
            info = (
                f"Model: Forge-1\n"
                f"Parameters: {n_params/1e6:.1f}M\n"
                f"Layers: {self.model.config.num_layers}\n"
                f"Hidden Dim: {self.model.config.hidden_dim}\n"
                f"Context Length: {self.model.config.context_length}\n"
                f"Device: {self.device}"
            )
            self.console.print(system_message(info))
        
        else:
            self.console.print(system_message(f"Unknown command: {cmd}"))
        
        return True
    
    def chat_turn(self, user_input: str):
        """Process a single chat turn."""
        from inference.thinking_pipeline import run_thinking_pipeline
        
        # Display user message
        self.console.print(user_message(user_input))
        
        # Show typing indicator
        with self.console.status("[bold green]Forge-1 is thinking...[/bold green]"):
            result = run_thinking_pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                user_message=user_input,
                max_new_tokens=512,
                temperature=0.7,
                device=self.device,
            )
        
        # Display response
        self.console.print(assistant_message(
            text=result.answer,
            thinking=result.thinking,
            show_thinking=self.show_thinking,
            gen_time=result.generation_time,
            num_tokens=result.num_tokens,
        ))
        
        # Store in history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": result.answer,
            "thinking": result.thinking,
            "tokens": result.num_tokens,
            "time": result.generation_time,
            "timestamp": datetime.now().isoformat(),
        })
    
    def run(self):
        """Main chat loop."""
        self.display_banner()
        
        while True:
            try:
                user_input = Prompt.ask("\n  [bold cyan]You[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                self.chat_turn(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /quit to exit[/dim]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                logger.exception("Chat error")
    
    def _save_conversation(self):
        """Save conversation to JSON file."""
        save_dir = Path("outputs/conversations")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"chat_{timestamp}.json"
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        
        self.console.print(system_message(f"Conversation saved to {save_path}"))
