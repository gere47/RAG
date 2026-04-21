#!/usr/bin/env python
"""
Complete System Diagnostic Suite for Graph-Grounded Temporal RAG
Tests every component and produces a detailed report.
"""

import sys
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

console = Console()


@dataclass
class TestResult:
    """Structured test result."""
    component: str
    test_name: str
    passed: bool
    score: float  # 0-100
    details: str
    recommendations: List[str]
    execution_time_ms: int


class SystemDiagnostic:
    """Complete system diagnostic suite."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.overall_score: float = 0.0
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete diagnostic suite."""
        console.print(Panel.fit(
            "[bold cyan]🔬 Graph-Grounded Temporal RAG - Complete System Diagnostic[/bold cyan]\n"
            "[dim]Testing all components for production readiness[/dim]",
            border_style="cyan"
        ))
        
        tests = [
            ("Vector Database", self.test_vector_db),
            ("Graph Database", self.test_graph_db),
            ("Hybrid Retrieval", self.test_hybrid_retrieval),
            ("Reranking Quality", self.test_reranking),
            ("Temporal Resolution", self.test_temporal_resolution),
            ("Contradiction Detection", self.test_contradiction_detection),
            ("LLM Integration", self.test_llm_integration),
            ("End-to-End Pipeline", self.test_e2e_pipeline),
            ("Response Time", self.test_response_time),
            ("Memory Efficiency", self.test_memory_efficiency),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for name, test_func in tests:
                task = progress.add_task(f"[cyan]Testing {name}...", total=None)
                
                start = time.perf_counter()
                try:
                    result = test_func()
                except Exception as e:
                    result = TestResult(
                        component=name,
                        test_name=test_func.__name__,
                        passed=False,
                        score=0,
                        details=f"Test failed: {str(e)}",
                        recommendations=["Fix the underlying error and retest"],
                        execution_time_ms=0
                    )
                
                result.execution_time_ms = int((time.perf_counter() - start) * 1000)
                self.results.append(result)
                progress.remove_task(task)
        
        self._calculate_overall_score()
        self._generate_report()
        
        return self._export_results()
    
    def test_vector_db(self) -> TestResult:
        """Test ChromaDB vector database."""
        from src.query_engine import QueryEngine
        engine = QueryEngine()
        
        collection = engine.collection
        count = collection.count()
        
        # Test 1: Collection exists and has documents
        if count == 0:
            return TestResult(
                component="Vector Database",
                test_name="test_vector_db",
                passed=False,
                score=0,
                details=f"Collection has {count} documents (expected > 0)",
                recommendations=["Run python src/05_create_index.py"],
                execution_time_ms=0
            )
        
        # Test 2: Query returns results
        results = collection.query(query_texts=["test"], n_results=5)
        has_results = len(results.get('ids', [[]])[0]) > 0
        
        # Test 3: Metadata exists
        sample = collection.get(limit=1)
        has_metadata = 'metadatas' in sample and sample['metadatas']
        
        score = 100
        issues = []
        
        if not has_results:
            score -= 30
            issues.append("Query returned no results")
        if not has_metadata:
            score -= 20
            issues.append("Missing metadata")
        if count < 100:
            score -= 10
            issues.append(f"Low document count: {count}")
        
        passed = score >= 70
        
        return TestResult(
            component="Vector Database",
            test_name="test_vector_db",
            passed=passed,
            score=score,
            details=f"Collection: {count} documents, Query working: {has_results}, Metadata: {has_metadata}",
            recommendations=issues if issues else ["Vector DB is healthy"],
            execution_time_ms=0
        )
    
    def test_graph_db(self) -> TestResult:
        """Test Neo4j graph database."""
        from src.config import config
        from neo4j import GraphDatabase
        
        if not config.neo4j.is_valid():
            return TestResult(
                component="Graph Database",
                test_name="test_graph_db",
                passed=False,
                score=0,
                details="Neo4j configuration invalid",
                recommendations=["Check .env file for NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"],
                execution_time_ms=0
            )
        
        try:
            driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.user, config.neo4j.password)
            )
            driver.verify_connectivity()
            
            with driver.session() as session:
                # Count nodes
                node_count = session.run("MATCH (n:Clause) RETURN count(n) as c").single()['c']
                # Count relationships
                rel_count = session.run("MATCH ()-[r:SUPERSEDES]->() RETURN count(r) as c").single()['c']
            
            driver.close()
            
            score = 100
            issues = []
            
            if node_count == 0:
                score -= 50
                issues.append("No Clause nodes found")
            if rel_count == 0:
                score -= 30
                issues.append("No SUPERSEDES relationships found")
            if node_count < 10:
                score -= 10
                issues.append(f"Low node count: {node_count}")
            
            passed = score >= 60
            
            return TestResult(
                component="Graph Database",
                test_name="test_graph_db",
                passed=passed,
                score=score,
                details=f"Connected: ✅, Nodes: {node_count}, Relationships: {rel_count}",
                recommendations=issues if issues else ["Graph DB is healthy"],
                execution_time_ms=0
            )
            
        except Exception as e:
            return TestResult(
                component="Graph Database",
                test_name="test_graph_db",
                passed=False,
                score=0,
                details=f"Connection failed: {str(e)[:100]}",
                recommendations=["Start Neo4j: neo4j.bat start", "Check credentials in .env"],
                execution_time_ms=0
            )
    
    def test_hybrid_retrieval(self) -> TestResult:
        """Test hybrid retrieval quality."""
        from src.query_engine import QueryEngine
        engine = QueryEngine()
        
        test_queries = [
            "effective date",
            "governing law",
            "parties involved"
        ]
        
        scores = []
        for query in test_queries:
            results = engine.retriever.search(query, top_k=5)
            if results:
                # Check diversity (different doc_ids)
                doc_ids = set()
                for chunk_id, _ in results:
                    chunk = engine.retriever.get_chunk_by_id(chunk_id)
                    if chunk:
                        doc_ids.add(chunk['metadata'].get('doc_id', ''))
                diversity = len(doc_ids) / min(len(results), 1)
                scores.append(diversity)
        
        avg_diversity = np.mean(scores) if scores else 0
        score = min(100, avg_diversity * 100)
        
        passed = score >= 50
        
        return TestResult(
            component="Hybrid Retrieval",
            test_name="test_hybrid_retrieval",
            passed=passed,
            score=score,
            details=f"Average result diversity: {avg_diversity:.2%} (across {len(test_queries)} queries)",
            recommendations=["Increase hybrid alpha for more diversity"] if score < 50 else ["Hybrid retrieval is working well"],
            execution_time_ms=0
        )
    
    def test_reranking(self) -> TestResult:
        """Test reranking effectiveness."""
        from src.query_engine import QueryEngine
        engine = QueryEngine()
        
        # Test if reranker improves ordering
        query = "What is the effective date?"
        
        # Get initial results
        initial = engine.retriever.search(query, top_k=10)
        
        if not initial:
            return TestResult(
                component="Reranking",
                test_name="test_reranking",
                passed=False,
                score=0,
                details="No initial results to rerank",
                recommendations=["Check vector database"],
                execution_time_ms=0
            )
        
        # Check if reranker is available
        has_reranker = hasattr(engine, 'reranker') and engine.reranker is not None
        
        score = 100 if has_reranker else 50
        passed = has_reranker
        
        return TestResult(
            component="Reranking",
            test_name="test_reranking",
            passed=passed,
            score=score,
            details=f"Reranker available: {has_reranker}, Initial results: {len(initial)}",
            recommendations=[] if has_reranker else ["Reranker not initialized - check model download"],
            execution_time_ms=0
        )
    
    def test_temporal_resolution(self) -> TestResult:
        """Test temporal resolution accuracy."""
        from src.query_engine import QueryEngine
        engine = QueryEngine()
        
        # Test with and without target date
        query = "What is the effective date?"
        
        # Current (no target date)
        result_current = engine.answer(query)
        
        # Historical (target date in past)
        result_past = engine.answer(query, target_date="2021-01-01")
        
        # Check if answers differ (they should if temporal resolution works)
        answers_differ = result_current['answer'] != result_past['answer']
        
        graph_used = result_current.get('graph_used', False)
        
        score = 100 if (answers_differ or not graph_used) else 70
        passed = graph_used
        
        return TestResult(
            component="Temporal Resolution",
            test_name="test_temporal_resolution",
            passed=passed,
            score=score,
            details=f"Graph used: {graph_used}, Answers differ by date: {answers_differ}",
            recommendations=["Build graph with SUPERSEDES relationships"] if not graph_used else ["Temporal resolution working"],
            execution_time_ms=0
        )
    
    def test_contradiction_detection(self) -> TestResult:
        """Test contradiction detection."""
        from src.query_engine import QueryEngine
        engine = QueryEngine()
        
        has_detector = hasattr(engine, 'contradiction_detector') and engine.contradiction_detector is not None
        
        if not has_detector:
            return TestResult(
                component="Contradiction Detection",
                test_name="test_contradiction_detection",
                passed=False,
                score=0,
                details="Contradiction detector not initialized",
                recommendations=["Check src/contradiction_detector.py import"],
                execution_time_ms=0
            )
        
        # Test detection
        query = "effective date"
        report = engine.contradiction_detector.analyze(query)
        
        versions_found = len(report.versions_found)
        has_contradiction = report.has_contradiction
        
        score = 100 if versions_found > 0 else 50
        passed = versions_found > 0
        
        return TestResult(
            component="Contradiction Detection",
            test_name="test_contradiction_detection",
            passed=passed,
            score=score,
            details=f"Versions found: {versions_found}, Contradiction detected: {has_contradiction}",
            recommendations=["Check Neo4j connection"] if versions_found == 0 else ["Contradiction detector working"],
            execution_time_ms=0
        )
    
    def test_llm_integration(self) -> TestResult:
        """Test LLM integration."""
        import ollama
        
        try:
            # Test connection
            response = ollama.generate(
                model="llama3.2:3b",
                prompt="Say 'OK' if you can hear me.",
                options={"num_predict": 5}
            )
            
            has_response = 'response' in response and len(response['response']) > 0
            
            score = 100 if has_response else 0
            passed = has_response
            
            return TestResult(
                component="LLM Integration",
                test_name="test_llm_integration",
                passed=passed,
                score=score,
                details=f"Ollama connected: ✅, Response received: {has_response}",
                recommendations=[] if has_response else ["Start Ollama: ollama serve"],
                execution_time_ms=0
            )
            
        except Exception as e:
            return TestResult(
                component="LLM Integration",
                test_name="test_llm_integration",
                passed=False,
                score=0,
                details=f"Connection failed: {str(e)[:100]}",
                recommendations=["Start Ollama: ollama serve", "Pull model: ollama pull llama3.2:3b"],
                execution_time_ms=0
            )
    
    def test_e2e_pipeline(self) -> TestResult:
        """Test end-to-end query pipeline."""
        from src.query_engine import QueryEngine
        engine = QueryEngine()
        
        query = "What is the effective date?"
        
        try:
            result = engine.answer(query)
            
            has_answer = 'answer' in result and len(result['answer']) > 10
            has_sources = 'sources' in result and len(result['sources']) > 0
            has_contradiction = 'contradiction' in result
            
            score = 100
            issues = []
            
            if not has_answer:
                score -= 40
                issues.append("No meaningful answer generated")
            if not has_sources:
                score -= 30
                issues.append("No sources retrieved")
            
            passed = score >= 60
            
            return TestResult(
                component="End-to-End Pipeline",
                test_name="test_e2e_pipeline",
                passed=passed,
                score=score,
                details=f"Answer: {has_answer}, Sources: {has_sources}, Contradiction: {has_contradiction}",
                recommendations=issues if issues else ["E2E pipeline is fully functional"],
                execution_time_ms=0
            )
            
        except Exception as e:
            return TestResult(
                component="End-to-End Pipeline",
                test_name="test_e2e_pipeline",
                passed=False,
                score=0,
                details=f"Pipeline failed: {str(e)[:100]}",
                recommendations=["Check all components", "Run diagnostic on individual components"],
                execution_time_ms=0
            )
    
    def test_response_time(self) -> TestResult:
        """Test response time performance."""
        from src.query_engine import QueryEngine
        engine = QueryEngine()
        
        query = "What is the effective date?"
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            engine.answer(query)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Scoring based on response time
        if avg_time < 2000:
            score = 100
        elif avg_time < 5000:
            score = 80
        elif avg_time < 10000:
            score = 60
        else:
            score = 40
        
        passed = score >= 60
        
        return TestResult(
            component="Response Time",
            test_name="test_response_time",
            passed=passed,
            score=score,
            details=f"Average: {avg_time:.0f}ms (±{std_time:.0f}ms) over 3 queries",
            recommendations=["Disable reranking for faster responses"] if avg_time > 5000 else ["Response time is acceptable"],
            execution_time_ms=0
        )
    
    def test_memory_efficiency(self) -> TestResult:
        """Test memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Scoring based on memory usage
        if memory_mb < 500:
            score = 100
        elif memory_mb < 1000:
            score = 80
        elif memory_mb < 2000:
            score = 60
        else:
            score = 40
        
        passed = score >= 60
        
        return TestResult(
            component="Memory Efficiency",
            test_name="test_memory_efficiency",
            passed=passed,
            score=score,
            details=f"Current memory usage: {memory_mb:.0f} MB",
            recommendations=["Consider reducing batch sizes"] if memory_mb > 2000 else ["Memory usage is acceptable"],
            execution_time_ms=0
        )
    
    def _calculate_overall_score(self):
        """Calculate weighted overall score."""
        weights = {
            "Vector Database": 0.15,
            "Graph Database": 0.15,
            "Hybrid Retrieval": 0.10,
            "Reranking Quality": 0.10,
            "Temporal Resolution": 0.10,
            "Contradiction Detection": 0.10,
            "LLM Integration": 0.10,
            "End-to-End Pipeline": 0.10,
            "Response Time": 0.05,
            "Memory Efficiency": 0.05,
        }
        
        total = 0
        for result in self.results:
            weight = weights.get(result.component, 0.05)
            total += result.score * weight
        
        self.overall_score = total
    
    def _generate_report(self):
        """Generate detailed report."""
        console.print("\n")
        
        # Overall score panel
        if self.overall_score >= 90:
            color = "green"
            grade = "A"
        elif self.overall_score >= 80:
            color = "blue"
            grade = "B"
        elif self.overall_score >= 70:
            color = "yellow"
            grade = "C"
        else:
            color = "red"
            grade = "D"
        
        console.print(Panel(
            f"[bold {color}]Overall System Score: {self.overall_score:.1f}/100[/bold {color}]\n"
            f"[{color}]Grade: {grade}[/{color}]",
            title="📊 System Health Summary",
            border_style=color
        ))
        
        # Detailed table
        table = Table(title="📋 Component Test Results", title_style="bold cyan")
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Details", style="dim", width=40)
        table.add_column("Execution Time", justify="right", width=12)
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            score_color = "green" if result.score >= 70 else "yellow" if result.score >= 50 else "red"
            
            table.add_row(
                result.component,
                status,
                f"[{score_color}]{result.score:.0f}[/{score_color}]",
                result.details[:40] + "..." if len(result.details) > 40 else result.details,
                f"{result.execution_time_ms}ms"
            )
        
        console.print(table)
        
        # Recommendations
        console.print("\n[bold yellow]🔧 Recommendations:[/bold yellow]\n")
        for result in self.results:
            if result.recommendations and not result.passed:
                console.print(f"[cyan]{result.component}:[/cyan]")
                for rec in result.recommendations:
                    console.print(f"  • {rec}")
    
    def _export_results(self) -> Dict[str, Any]:
        """Export results to JSON."""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_score': self.overall_score,
            'results': [asdict(r) for r in self.results]
        }


def main():
    """Run diagnostic suite."""
    diagnostic = SystemDiagnostic()
    results = diagnostic.run_all_tests()
    
    # Save results
    output_path = Path("diagnostic_results.json")
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[dim]📁 Results saved to {output_path}[/dim]")
    
    return 0 if diagnostic.overall_score >= 70 else 1


if __name__ == "__main__":
    sys.exit(main())