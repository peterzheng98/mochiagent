#!/usr/bin/env python3
"""
Demo script for MochiAgent - Medical Agent System

This script demonstrates the full agent pipeline with sample data.
It can be run with or without the MCP servers (uses fallback mode).
"""

import os
import sys
import json
import argparse
import threading
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import generate_task_id, gprint, yprint, bprint, rprint


def start_servers():
    """Start MCP servers in background threads."""
    from server.gpt_server import GPTServer
    from server.transformer_server import TransformerMCPServer
    from server.websearch_server import WebSearchMCPServer
    from server.trajectory_server import TrajectoryMCPServer
    from server.clustering_server import ClusteringMCPServer
    
    servers = [
        ("GPT", GPTServer),
        ("Transformer", TransformerMCPServer),
        ("WebSearch", WebSearchMCPServer),
        ("Trajectory", TrajectoryMCPServer),
        ("Clustering", ClusteringMCPServer),
    ]
    
    threads = []
    for name, ServerClass in servers:
        try:
            server = ServerClass()
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            threads.append(thread)
            gprint(f"Started {name} server")
        except Exception as e:
            yprint(f"Could not start {name} server: {e}")
    
    return threads


def get_sample_data(task_type: str) -> Dict[str, Any]:
    """Get sample data for different task types."""
    samples = {
        "diabetes": {
            "ehr": [
                "Patient is a 45-year-old male with history of Type 2 Diabetes Mellitus.",
                "Presenting with fatigue, polyuria, and polydipsia for 2 weeks.",
                "BMI 28.5, Blood pressure 140/90 mmHg.",
                "Family history positive for diabetes and cardiovascular disease."
            ],
            "lab_tests": [
                [126, 7.2, 180, 45, 150],  # FBS, HbA1c, LDL, HDL, TG
                [135, 7.5, 175, 42, 160],
                [142, 7.8, 185, 40, 170],
            ]
        },
        "cardiac": {
            "ehr": [
                "65-year-old female presenting with chest pain and shortness of breath.",
                "History of hypertension and hyperlipidemia.",
                "ECG shows ST elevation in leads V1-V4.",
                "Previous MI 5 years ago with stent placement."
            ],
            "lab_tests": [
                [0.5, 250, 180, 35, 200, 12.5],  # Troponin, CK-MB, LDL, HDL, TG, BNP
                [2.5, 450, 175, 34, 195, 850],
                [4.2, 380, 172, 33, 190, 1200],
            ]
        },
        "renal": {
            "ehr": [
                "58-year-old male with chronic kidney disease stage 3.",
                "Diabetes and hypertension for 15 years.",
                "Presenting with edema and decreased urine output.",
                "Currently on ACE inhibitor and statin therapy."
            ],
            "lab_tests": [
                [2.1, 45, 6.2, 140, 4.8, 22],  # Creatinine, eGFR, Urea, Na, K, HCO3
                [2.4, 38, 7.5, 138, 5.2, 20],
                [2.8, 32, 8.8, 136, 5.5, 18],
            ]
        },
        "default": {
            "ehr": [
                "Patient 45M, history of T2DM.",
                "Presenting with fatigue and polyuria."
            ],
            "lab_tests": [
                [100, 20, 30],
                [110, 22, 32],
                [105, 21, 31]
            ]
        }
    }
    
    return samples.get(task_type, samples["default"])


def run_demo(task_type: str, start_server: bool = False):
    """Run the demo with specified task type."""
    bprint("=" * 60)
    bprint("MochiAgent - Medical Agent System Demo")
    bprint("=" * 60)
    
    # Start servers if requested
    if start_server:
        bprint("Starting MCP servers...")
        start_servers()
        import time
        time.sleep(2)  # Wait for servers to initialize
    
    # Get sample data
    data = get_sample_data(task_type)
    
    bprint(f"Task Type: {task_type}")
    bprint(f"EHR Records: {len(data['ehr'])}")
    bprint(f"Lab Test Rows: {len(data['lab_tests'])}")
    bprint("-" * 60)
    
    # Initialize agent
    from agent import MedicalAgent
    
    config = Config(generate_task_id())
    agent = MedicalAgent(config)
    
    # Progress callback for demo
    def progress_callback(step, status, details):
        if status == "running":
            bprint(f"  [{step}] Starting...")
        elif status == "completed":
            gprint(f"  [{step}] Completed")
    
    # Run analysis
    bprint("Running Analysis...")
    results = agent.run(
        data["ehr"],
        data["lab_tests"],
        progress_callback=progress_callback
    )
    
    # Display results
    bprint("-" * 60)
    bprint("RESULTS")
    bprint("-" * 60)
    
    # Transformer prediction
    if "transformer_prediction" in results:
        pred = results["transformer_prediction"].get("prediction", {})
        bprint("Transformer Prediction:")
        print(f"  Risk Score: {pred.get('prediction_score', 'N/A'):.3f}")
        print(f"  Risk Category: {pred.get('risk_category', 'N/A')}")
        print(f"  Confidence: {pred.get('confidence', 'N/A'):.2f}")
    
    # Reasoning
    if "reasoning" in results:
        bprint("Web Search Reasoning:")
        reasoning = results["reasoning"].get("reasoning", "N/A")
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."
        print(f"  {reasoning}")
    
    # Trajectory
    if "trajectory_inference" in results:
        traj = results["trajectory_inference"]
        bprint("Trajectory Inference:")
        graph = traj.get("graph", {})
        print(f"  Nodes: {len(graph.get('nodes', []))}")
        print(f"  Edges: {len(graph.get('edges', []))}")
    
    # Clustering
    if "clustering" in results:
        clust = results["clustering"]
        bprint("Clustering Analysis:")
        graph = clust.get("graph", {})
        print(f"  Clusters: {len(graph.get('clusters', {}))}")
        metrics = clust.get("quality_metrics", {})
        print(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.3f}")
    
    bprint("=" * 60)
    gprint("Demo Complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MochiAgent Demo')
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['diabetes', 'cardiac', 'renal', 'default'],
        default='default',
        help='Task type for demo'
    )
    parser.add_argument(
        '--start-servers',
        action='store_true',
        help='Start MCP servers before running demo'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    results = run_demo(args.task, args.start_servers)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        gprint(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

