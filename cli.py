
import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup working directory to project root if not already
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    clear_screen()
    rprint(Panel(
        "[bold cyan]AetherMind AI Manager[/bold cyan]\n"
        "[dim]Interactive Command Center for Forge-1 Models[/dim]",
        border_style="cyan",
        expand=False
    ))

def get_datasets():
    # Search common locations
    candidates = []
    
    # Check default data directory
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        candidates.extend(data_dir.glob("*.jsonl"))
        
    return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)

def get_models():
    # Check common output locations
    candidates = []
    
    # Recursively find .pt files in outputs/
    output_dir = PROJECT_ROOT / "outputs"
    if output_dir.exists():
        # Look in outputs/final_model, outputs/models, and just anywhere in outputs
        # We limit depth to avoid deep checkpoint structures if possible, but finding all .pt is good
        for path in output_dir.rglob("*.pt"):
            # Exclude optimizer states if they are saved separately (usually same file though)
            # Filter out temporary checkpoints if desired, but user might want them
            candidates.append(path)
        
    return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)

def train_flow():
    print_banner()
    rprint("[bold yellow]Configuration Wizard: New Training Run[/bold yellow]\n")

    # --- Model Configuration ---
    rprint("[bold]1. Model Settings[/bold]")
    model_name = Prompt.ask("Enter a name for your model", default="my-forge-model")
    variant = Prompt.ask("Select model variant", choices=["nano", "mini"], default="nano")
    
    # Model Save Location
    default_save_dir = PROJECT_ROOT / "outputs" / "models"
    save_dir_input = Prompt.ask(
        "Directory to save the trained model", 
        default=str(default_save_dir)
    )
    model_save_dir = Path(save_dir_input)
    
    # --- Dataset Configuration ---
    rprint("\n[bold]2. Dataset Selection[/bold]")
    dataset_choice = Prompt.ask("Do you want to use an existing dataset or create a new one?", 
                              choices=["existing", "create"], default="existing")
    
    dataset_path = None
    
    if dataset_choice == "create":
        rprint("\n[dim]Creating a new dataset...[/dim]")
        
        # Dataset Location
        default_data_dir = PROJECT_ROOT / "data"
        data_dir_input = Prompt.ask(
            "Directory to save the new dataset", 
            default=str(default_data_dir)
        )
        data_save_dir = Path(data_dir_input)
        
        # Ensure dir exists
        if not data_save_dir.exists():
            if Confirm.ask(f"Directory {data_save_dir} does not exist. Create it?"):
                data_save_dir.mkdir(parents=True, exist_ok=True)
            else:
                return

        dataset_name = Prompt.ask("Name for the new dataset file (e.g. custom.jsonl)", default="custom.jsonl")
        if not dataset_name.endswith(".jsonl"):
            dataset_name += ".jsonl"
            
        max_samples = IntPrompt.ask("Max samples to use per source dataset (e.g. 1000 => 3000 total if 3 sources)", default=1000)
        
        output_path = data_save_dir / dataset_name
        
        # Check overwrite
        if output_path.exists():
            if not Confirm.ask(f"{output_path} already exists. Overwrite?"):
                return
        
        # Run preprocessing
        cmd = [
            sys.executable, 
            str(PROJECT_ROOT / "data" / "preprocess.py"),
            "--max-samples", str(max_samples),
            "--output", str(output_path)
        ]
        
        rprint(f"\n[green]Running preprocessing...[/green]")
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
                progress.add_task(description="Downloading and processing dataset...", total=None)
                result = subprocess.run(cmd, capture_output=False, check=True)
            rprint(f"[green]Dataset created at: {output_path}[/green]")
            dataset_path = output_path
        except subprocess.CalledProcessError:
            rprint("[bold red]Error during dataset creation![/bold red]")
            Prompt.ask("Press Enter to return to menu...")
            return
        
    else:
        # Existing dataset
        datasets = get_datasets()
        
        # Allow user to specify custom path if not found
        if not datasets:
            rprint("[yellow]No .jsonl datasets found in default locations.[/yellow]")
        
        rprint("\n[cyan]Option A: Select from found datasets:[/cyan]")
        for idx, ds in enumerate(datasets):
            rprint(f"  {idx+1}. {ds.name} ({ds.stat().st_size / 1024 / 1024:.2f} MB) in {ds.parent}")
            
        rprint(f"  {len(datasets)+1}. Enter custom path")
        
        ds_choice = IntPrompt.ask("Select option", choices=[str(i+1) for i in range(len(datasets)+1)])
        
        if ds_choice == len(datasets) + 1:
            custom_path = Prompt.ask("Enter full path to .jsonl file")
            dataset_path = Path(custom_path)
            if not dataset_path.exists():
                rprint(f"[red]File not found: {dataset_path}[/red]")
                Prompt.ask("Press Enter to return...")
                return
        else:
            dataset_path = datasets[ds_choice-1]

    # --- Training Execution ---
    rprint(f"\n[bold green]Ready to train '{model_name}' ({variant})[/bold green]")
    rprint(f"Dataset: {dataset_path}")
    rprint(f"Output:  {model_save_dir / model_name}.pt")
    
    if not Confirm.ask("Start training now?"):
        return

    # Create model save directory
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True, exist_ok=True)

    # Construct training command
    config_path = PROJECT_ROOT / "configs" / "local_config.yaml"
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--config", str(config_path),
        "--variant", variant,
        "--data-path", str(dataset_path)
    ]
    
    rprint("\n[dim]Initializing training process... (This may take a moment)[/dim]")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        rprint("[bold red]Training failed![/bold red]")
        Prompt.ask("Press Enter to return to menu...")
        return
        
    # --- Post-Training File Handling ---
    # Default output from train.py is outputs/final_model/forge1_<variant>_final.pt
    # and outputs/final_model/tokenizer/
    
    default_output_dir = PROJECT_ROOT / "outputs" / "final_model"
    source_model_file = default_output_dir / f"forge1_{variant}_final.pt"
    source_tokenizer_dir = default_output_dir / "tokenizer"
    
    target_model_file = model_save_dir / f"{model_name}.pt"
    # We might want to save the tokenizer alongside.
    # If the user specified a custom directory, maybe they want the tokenizer there too.
    # Let's put it in a subdirectory named {model_name}_tokenizer or similar if possible, 
    # or just "tokenizer" if the directory is exclusive to this model.
    # Simplest approach: "tokenizer" folder in the save dir.
    target_tokenizer_dir = model_save_dir / "tokenizer"

    if source_model_file.exists():
        try:
            # Move/Rename Model
            rprint(f"Moving model to {target_model_file}...")
            if target_model_file.exists():
                 # backup or overwrite? Overwrite for now as it's a new run
                 target_model_file.unlink()
            shutil.move(str(source_model_file), str(target_model_file))
            
            # Move/Copy Tokenizer
            if source_tokenizer_dir.exists():
                rprint(f"Copying tokenizer to {target_tokenizer_dir}...")
                if target_tokenizer_dir.exists():
                    shutil.rmtree(str(target_tokenizer_dir))
                shutil.copytree(str(source_tokenizer_dir), str(target_tokenizer_dir))
                
            rprint(f"\n[bold green]Model and artifacts saved successfully![/bold green]")
            
        except Exception as e:
            rprint(f"[red]Error moving files: {e}[/red]")
            # If move failed, they are still in default location
            target_model_file = source_model_file 
    else:
        rprint(f"[yellow]Warning: Could not find expected output file {source_model_file}. Check logs.[/yellow]")

    # --- Review Options ---
    if Confirm.ask("Do you want to chat with this model now?"):
        run_chat(target_model_file, variant)
    elif Confirm.ask("Do you want to evaluate this model?"):
        run_eval(target_model_file, variant)

def run_chat(model_path, variant):
    # Check if tokenizer is alongside the model
    # Chat script looks for tokenizer arg or defaults.
    # If we moved tokenizer to model_dir/tokenizer, we should pass that.
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "chat.py"),
        "--model", str(model_path),
        "--variant", variant
    ]
    
    # Try to find tokenizer
    model_dir = Path(model_path).parent
    possible_tokenizer = model_dir / "tokenizer"
    if possible_tokenizer.exists():
        cmd.extend(["--tokenizer", str(possible_tokenizer)])
        
    subprocess.run(cmd)
    Prompt.ask("\nChat session ended. Press Enter to continue...")

def run_eval(model_path, variant):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--model", str(model_path),
        "--variant", variant
    ]
    subprocess.run(cmd)
    Prompt.ask("\nEvaluation finished. Press Enter to continue...")

def select_model_flow(action="chat"):
    print_banner()
    models = get_models()
    
    if not models:
        rprint("[red]No trained models found recursively in outputs/![/red]")
        Prompt.ask("Press Enter to return...")
        return

    rprint(f"[bold]Select a model to {action}:[/bold]")
    for idx, model in enumerate(models):
        # Show relative path if possible for brevity
        try:
            display_name = model.relative_to(PROJECT_ROOT)
        except ValueError:
            display_name = model
            
        rprint(f"  {idx+1}. {display_name}")
    
    choice = IntPrompt.ask("Choice", choices=[str(i+1) for i in range(len(models))])
    model_path = models[choice-1]
    
    # Guess variant from name if possible, else ask
    variant = "nano"
    if "mini" in model_path.name.lower():
        variant = "mini"
    
    variant = Prompt.ask("Confirm variant", choices=["nano", "mini"], default=variant)
    
    if action == "chat":
        run_chat(model_path, variant)
    else:
        run_eval(model_path, variant)

def main():
    try:
        while True:
            print_banner()
            rprint("\n[bold]Main Menu[/bold]")
            rprint("1. [green]Train New Model[/green]")
            rprint("2. [blue]Chat with Model[/blue]")
            rprint("3. [magenta]Evaluate Model[/magenta]")
            rprint("4. [red]Exit[/red]")
            
            choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4"])
            
            if choice == "1":
                try:
                    train_flow()
                except KeyboardInterrupt:
                    continue
            elif choice == "2":
                try:
                    select_model_flow(action="chat")
                except KeyboardInterrupt:
                    continue
            elif choice == "3":
                try:
                    select_model_flow(action="eval")
                except KeyboardInterrupt:
                    continue
            elif choice == "4":
                rprint("[dim]Goodbye![/dim]")
                sys.exit(0)
    except KeyboardInterrupt:
        rprint("\n[dim]Goodbye![/dim]")
        sys.exit(0)

if __name__ == "__main__":
    main()
