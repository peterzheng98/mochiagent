import typer
import json
import os
from agent.core import Agent
from typing import Optional

app = typer.Typer()

@app.command()
def run(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to JSON input file containing 'ehr' and 'lab_tests'"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save the output JSON result")
):
    """
    Run the agent using a JSON input file.
    
    The input file should be a JSON object with:
    - "ehr": List[str]
    - "lab_tests": List[List[Any]]
    """
    if not os.path.exists(input_file):
        typer.echo(f"Error: Input file '{input_file}' not found.")
        raise typer.Exit(code=1)

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        typer.echo(f"Error: Failed to parse '{input_file}'. Ensure it is a valid JSON file.")
        raise typer.Exit(code=1)

    if "ehr" not in data or "lab_tests" not in data:
        typer.echo("Error: Input JSON must contain 'ehr' and 'lab_tests' keys.")
        raise typer.Exit(code=1)

    ehr_list = data["ehr"]
    lab_tests = data["lab_tests"]

    typer.echo("Initializing Agent...")
    agent = Agent()
    
    typer.echo("Running Agent...")
    results = agent.run(ehr_list, lab_tests)

    typer.echo("Analysis Complete.")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        typer.echo(f"Results saved to {output_file}")
    else:
        typer.echo(json.dumps(results, indent=2))

if __name__ == "__main__":
    app()

