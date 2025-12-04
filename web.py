#!/usr/bin/env python3
"""
Web Interface for MochiAgent
"""

import os
import sys
import json
import threading
import queue
from typing import List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from agent import MedicalAgent
from config import Config
from utils import generate_task_id


app = FastAPI(
    title="MochiAgent Web Interface",
    description="Medical Agent System for EHR and Lab Test Analysis",
    version="1.0.0"
)

# Templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""
    ehr: List[str]
    lab_tests: List[List[Any]]


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """
    Analyze EHR and lab test data.
    
    Returns a stream of JSON objects (NDJSON) with progress updates.
    """
    q: queue.Queue = queue.Queue()
    
    def progress_callback(step, status, details):
        q.put({
            "type": "progress",
            "step": step,
            "status": status,
            "details": details
        })
    
    def run_agent_thread():
        try:
            config = Config(generate_task_id())
            agent = MedicalAgent(config)
            results = agent.run(
                request.ehr,
                request.lab_tests,
                progress_callback=progress_callback
            )
            q.put({
                "type": "result",
                "data": results
            })
        except Exception as e:
            q.put({
                "type": "error",
                "message": str(e)
            })
        finally:
            q.put(None)  # Sentinel
    
    thread = threading.Thread(target=run_agent_thread)
    thread.start()
    
    def event_generator():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, default=str) + "\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson"
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mochiagent"}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>MochiAgent</title></head>
        <body>
            <h1>MochiAgent API</h1>
            <p>Web interface templates not found.</p>
            <p>Use POST /api/analyze to submit analysis requests.</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
        </html>
        """


def main():
    """Run the web server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MochiAgent Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "web:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

