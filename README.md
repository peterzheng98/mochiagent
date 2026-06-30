# MochiAgent

MochiAgent is an advanced medical agent system designed to analyze Electronic Health Records (EHR) and laboratory test data. It implements a Model Context Protocol (MCP) server architecture with multiple specialized components including transformer-based inference, web search for reasoning, single-cell trajectory inference, and clustering analysis.

## Architecture

The system follows a microservices architecture with MCP servers:

```
MochiAgent/
    agent.py              # Main agent orchestrator
    config.py             # Configuration management
    utils.py              # Utility functions
    demo.py               # Demo script
    cli.py                # Command-line interface
    web.py                # Web interface (FastAPI)
    server/               # MCP Server implementations
        base.py           # Base server classes
        transformer_server.py   # serves the MoChiFormer engine (see mochiformer/)
        websearch_server.py
        trajectory_server.py
        clustering_server.py
        gpt_server.py
        code_executor.py
    mochiformer/          # MoChiFormer core model (trainable; real, not placeholder)
        config.py         # MoChiFormerConfig + demo_config
        model.py          # BERT visit-encoder + GPT-2 temporal decoder + heads
        data.py           # discretization, tokenizer, dataset, synthetic cohort
        train.py          # pretrain + finetune + checkpoint I/O (CLI: `demo`)
        inference.py      # MoChiFormerPredictor (load -> predict) + raw adapter
        README.md         # paper<->code mapping, scope, quickstart
    checkpoints/          # trained MoChiFormer checkpoints (e.g. mochiformer_demo.ckpt)
    scripts/              # Core scripts
        chat.py           # Chat session management
        component.py      # Task and status components
        llm.py            # LLM call utilities
        prompt.py         # Prompt templates
    tool/                 # Tool definitions
        doc/              # Tool documentation
        code/             # Tool implementations
    templates/            # Web interface templates
    tests/                # Smoke test (train -> infer -> serve)
    data/                 # Sample data
```

## Features

- **Transformer Inference (MoChiFormer)**: A real, trainable longitudinal-EHR transformer (BERT visit-encoder + GPT-2 temporal decoder) that predicts disease risk and biological age from discretized lab + EHR sequences. See [`mochiformer/`](mochiformer/README.md).
- **Web Search & Reasoning**: DuckDuckGo integration for medical information retrieval and reasoning
- **Trajectory Inference**: Single-cell trajectory analysis with pseudotime computation
- **Clustering**: Multiple algorithms (K-means, Leiden, Hierarchical) for pattern discovery
- **MCP Protocol**: Standardized server communication with tools, resources, and prompts
- **Dual Interfaces**: CLI for batch processing and Web UI for interactive analysis

## MoChiFormer (core model)

The transformer tool is backed by **MoChiFormer**, the longitudinal-EHR
foundation model described in the paper (Methods, "The core prediction model:
MoChiFormer"). It lives in [`mochiformer/`](mochiformer/README.md) and is a real,
trainable PyTorch model — a BERT visit-encoder + GPT-2 temporal decoder with
optional VAE/KL and cohort-adversarial de-biasing, focal-loss multi-disease
heads, and age regression — operating on discretized lab + structured-EHR
sequences.

It supports the full train → infer → serve loop:

```bash
# from the mochiagent/ directory
PY=python   # use a Python env with torch + transformers installed

# 1. Train a self-contained synthetic demo -> checkpoints/mochiformer_demo.ckpt
$PY -m mochiformer.train demo --out checkpoints/mochiformer_demo.ckpt

# 2. End-to-end smoke test (train -> infer -> serve)
$PY tests/test_smoke.py
```

At serve time, `server/transformer_server.py` automatically loads
`checkpoints/mochiformer_demo.ckpt` (override with `MOCHIFORMER_CKPT=/path.ckpt`)
and runs real inference through the `predict` tool. If torch or a checkpoint is
unavailable it degrades to a clearly-labelled placeholder so the rest of the
system still runs.

Extra dependencies beyond the base `requirements.txt`:
`torch`, `transformers` (see [`mochiformer/requirements.txt`](mochiformer/requirements.txt)).

> The bundled demo checkpoint is trained on **synthetic** data — it demonstrates
> that the code is correct and runnable, not clinical performance. Train on real,
> schema-matched data for meaningful predictions. See
> [`mochiformer/README.md`](mochiformer/README.md) for the paper↔code mapping,
> how to plug in real data, and what is intentionally out of scope.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mochiagent
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Start Redis for MCP server communication:
```bash
redis-server
```

## Usage

### Command Line Interface

Run analysis on a JSON input file:
```bash
python cli.py run --input sample_input.json --output results.json
```

Start an MCP server:
```bash
python cli.py server transformer
python cli.py server websearch
python cli.py server trajectory
python cli.py server clustering
```

### Web Interface

Start the web server:
```bash
python web.py --port 8000
```

Then open http://localhost:8000 in your browser.

## Input Format

The agent expects JSON input with two required fields:

```json
{
    "ehr": [
        "Patient 45M, history of T2DM."
    ],
    "lab_tests": [
        [126, 7.2, 180, 45, 150],
        [135, 7.5, 175, 42, 160]
    ]
}
```

- `ehr`: List of EHR text records (strings)
- `lab_tests`: 2D array of lab test values (rows = time points, columns = test types)

## MCP Server Protocol

Each MCP server implements:

- **Tools**: Callable functions for specific tasks
- **Resources**: Data and configuration access
- **Prompts**: Template prompts for LLM interactions

Example tool call:
```python
from server.transformer_server import TransformerMCPServer

server = TransformerMCPServer()
result = server.call_tool("predict", {
    "ehr_data": ["Patient data..."],
    "lab_tests": [[100, 20, 30]]
})
```

## Configuration

Configuration is managed in `config.py`:

- LLM model settings
- Redis connection parameters
- Server ports and timeouts
- Task processing parameters

Environment variables:
- `OPENAI_API_KEY`: OpenAI API key for LLM calls
- `OPENAI_BASE_URL`: Custom API endpoint

## API Endpoints

- `POST /api/analyze`: Submit analysis request (streaming NDJSON response)
- `GET /api/health`: Health check
- `GET /docs`: API documentation (Swagger UI)

## Output Format

Analysis results include:

```json
{
    "transformer_prediction": {
        "prediction": {
            "prediction_score": 0.75,
            "risk_category": "MODERATE",
            "confidence": 0.85
        }
    },
    "reasoning": {
        "query": "...",
        "reasoning": "...",
        "search_results": [...]
    },
    "trajectory_inference": {
        "graph": {...},
        "pseudotime": {...}
    },
    "clustering": {
        "labels": [...],
        "clusters": {...},
        "quality_metrics": {...}
    }
}
```

## Development

Run tests:
```bash
python -m pytest tests/
```

Start development server with auto-reload:
```bash
python web.py --reload
```

## Previous version

The previous old committed version of this repository is available on GitHub at commit. You can use it if it contains compatiability problem.
[`2ccaba6`](https://github.com/peterzheng98/mochiagent/commit/2ccaba63746754ae0e524b5a2bc7e6369f45d67f).


## License

MIT License
