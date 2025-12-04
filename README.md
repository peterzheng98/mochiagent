# MochiAgent

MochiAgent is an advanced medical agent designed to analyze Electronic Health Records (EHR) and lab test data. It integrates multiple components including a transformer-based inference engine, web search for reasoning, single-cell trajectory inference, and clustering analysis to provide comprehensive insights.

**Note:** This project adheres to a strict "No Emoji" policy.

## Features

- **Transformer Inference:** Predicts results based on EHR and lab test data.
- **Web Search & Reasoning:** Performs web searches to provide context and reasoning for the inference results.
- **Trajectory Inference:** Simulates single-cell trajectory analysis on lab test data.
- **Clustering:** Performs clustering analysis on patient data.
- **Dual Interfaces:**
    - **CLI:** Command-line interface for batch processing and automation.
    - **Web UI:** Modern, responsive web interface with real-time process visualization and streaming updates.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mochiagent
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface (CLI)

You can run the agent directly from the terminal using `cli.py`.

**Command:**
```bash
python cli.py --input <path_to_input_json> [--output <path_to_output_json>]
```

**Example:**
```bash
python cli.py --input sample_input.json --output results.json
```

### Web Interface

The web interface provides a user-friendly way to interact with the agent and visualize the analysis process step-by-step.

1. Start the web server:
   ```bash
   python web.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Enter your JSON data in the input area and click "Run Analysis".

## Input Format

The agent expects input data in JSON format with two required keys: `ehr` and `lab_tests`.

**Structure:**
```json
{
    "ehr": [
        "String containing patient history",
        "String containing symptoms"
    ],
    "lab_tests": [
        [value1, value2, ...],
        [value1, value2, ...]
    ]
}
```

**Example:**
```json
{
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
```

## Project Structure

- `agent/`: Core agent logic and components.
    - `core.py`: Main agent orchestrator.
    - `transformer.py`: Transformer inference engine.
    - `web_search.py`: Web search and reasoning tool.
    - `trajectory.py`: Trajectory inference wrapper.
    - `clustering.py`: Clustering analysis wrapper.
- `templates/`: HTML templates for the web interface.
- `cli.py`: Command-line interface entry point.
- `web.py`: FastAPI web server entry point.
- `requirements.txt`: Python dependencies.

