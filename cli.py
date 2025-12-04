#!/usr/bin/env python3
"""
Command Line Interface for MochiAgent
"""

import os
import sys
import json
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_analysis(input_file: str, output_file: Optional[str] = None):
    """Run the agent analysis on input data."""
    from agent import MedicalAgent
    from config import Config
    from utils import generate_task_id, gprint, rprint
    
    # Load input data
    if not os.path.exists(input_file):
        rprint(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        rprint(f"Error: Failed to parse '{input_file}': {e}")
        sys.exit(1)
    
    # Validate input
    if "ehr" not in data or "lab_tests" not in data:
        rprint("Error: Input JSON must contain 'ehr' and 'lab_tests' keys.")
        sys.exit(1)
    
    ehr_list = data["ehr"]
    lab_tests = data["lab_tests"]
    
    print("Initializing Agent...")
    config = Config(generate_task_id())
    agent = MedicalAgent(config)
    
    # Progress callback
    def progress_callback(step, status, details):
        if status == "running":
            print(f"  [{step}] Processing...")
        elif status == "completed":
            print(f"  [{step}] Done")
    
    print("Running Analysis...")
    results = agent.run(ehr_list, lab_tests, progress_callback=progress_callback)
    
    print("Analysis Complete.")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        gprint(f"Results saved to {output_file}")
    else:
        print(json.dumps(results, indent=2, default=str))
    
    return results


def start_server(server_type: str):
    """Start a specific MCP server."""
    from utils import gprint
    
    server_map = {
        "gpt": ("server.gpt_server", "GPTServer"),
        "transformer": ("server.transformer_server", "TransformerMCPServer"),
        "websearch": ("server.websearch_server", "WebSearchMCPServer"),
        "trajectory": ("server.trajectory_server", "TrajectoryMCPServer"),
        "clustering": ("server.clustering_server", "ClusteringMCPServer"),
        "executor": ("server.code_executor", "CodeExecutorServer"),
    }
    
    if server_type not in server_map:
        print(f"Unknown server type: {server_type}")
        print(f"Available: {', '.join(server_map.keys())}")
        sys.exit(1)
    
    module_name, class_name = server_map[server_type]
    
    import importlib
    module = importlib.import_module(module_name)
    ServerClass = getattr(module, class_name)
    
    gprint(f"Starting {server_type} server...")
    server = ServerClass()
    server.run()


def main():
    parser = argparse.ArgumentParser(
        description='MochiAgent - Medical Agent CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run analysis:
    python cli.py run --input sample_input.json --output results.json
  
  Start a server:
    python cli.py server transformer
    python cli.py server websearch
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run agent analysis')
    run_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to JSON input file'
    )
    run_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save output JSON'
    )
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start an MCP server')
    server_parser.add_argument(
        'server_type',
        type=str,
        choices=['gpt', 'transformer', 'websearch', 'trajectory', 'clustering', 'executor'],
        help='Type of server to start'
    )
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_analysis(args.input, args.output)
    elif args.command == 'server':
        start_server(args.server_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

