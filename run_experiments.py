#!/usr/bin/env python
"""
One-command reproduction of all experiments.
Generates publication-ready figures and tables.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_command(cmd: str, description: str) -> bool:
    """Run a command with nice output."""
    console.print(f"\n[bold cyan]{description}[/bold cyan]")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    console.print(Panel.fit(
        "[bold green]🔬 Graph-Grounded Temporal RAG - Full Experiment Reproduction[/bold green]\n"
        "[dim]This will reproduce all results from the paper[/dim]",
        border_style="green"
    ))
    
    steps = [
        ("python run_pipeline.py --clean", "Step 1: Building indexes and graph"),
        ("python evaluation/evaluate.py", "Step 2: Running evaluation suite"),
        ("python demo.py --export --quiet", "Step 3: Generating demo results"),
        ("python -c 'from src.query_engine import QueryEngine; print(\"System ready\")'", "Step 4: Verification"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            console.print(f"[red]❌ Failed: {desc}[/red]")
            return 1
    
    console.print("\n[bold green]✅ All experiments reproduced successfully![/bold green]")
    console.print("[dim]Results available in:[/dim]")
    console.print("  - evaluation/results/")
    console.print("  - demo_results.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())