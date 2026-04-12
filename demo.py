"""
One-click demo script for showcasing Graph-Grounded Temporal RAG.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_engine import QueryEngine
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
console = Console()


def run_demo():
    console.print(Panel.fit(
        "[bold cyan]⚖️ Graph-Grounded Temporal RAG[/bold cyan]\n"
        "[dim]Contradiction-Resilient QA over Evolving Legal Documents[/dim]",
        border_style="cyan"
    ))
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Initializing system...", total=100)
        
        for i in range(100):
            progress.update(task, advance=1)
        
        engine = QueryEngine()
    
    # System Status Table
    table = Table(title="📊 System Status", title_style="bold cyan")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    table.add_row("Vector DB", "✅ Online", f"{engine.collection.count()} chunks")
    table.add_row("Graph DB", "✅ Connected" if engine.graph_enabled else "⚠️ Disabled", 
                  "Neo4j" if engine.graph_enabled else "Vector-only mode")
    table.add_row("LLM Model", "✅ Ready", engine.config.ollama.model)
    table.add_row("Reranker", "✅ Ready", "Cross-Encoder (MS MARCO)")
    
    console.print(table)
    
    # Demo Queries
    demo_questions = [
        ("📄 Current State", "What is the effective date of the original agreement?"),
        ("⏳ Temporal Query", "What was the fee in 2021?"),
        ("🔄 Contradiction Test", "Has the penalty clause changed over time?"),
        ("🔍 Specific Clause", "What does Section 1.1 state?"),
    ]
    
    console.print("\n[bold yellow]🎯 Running Demo Queries...[/bold yellow]\n")
    
    for category, question in demo_questions:
        console.print(f"[bold cyan]{category}:[/bold cyan] {question}")
        
        with Progress() as progress:
            task = progress.add_task("[dim]Retrieving...", total=100)
            result = engine.answer(question)
            progress.update(task, completed=100)
        
        console.print(f"[green]Answer:[/green] {result['answer'][:200]}...")
        console.print(f"[dim]Sources: {len(result['sources'])} | Graph used: {result['graph_used']}[/dim]")
        console.print("─" * 60)
    
    console.print("\n[bold green]✅ Demo Complete![/bold green]")


if __name__ == "__main__":
    run_demo()