from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Any
from agent.core import Agent
import uvicorn
import os
import json
import threading
import queue

app = FastAPI(title="Agent Web Interface")

# Mount templates directory if it exists, otherwise we'll serve a simple string
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

class AnalysisRequest(BaseModel):
    ehr: List[str]
    lab_tests: List[List[Any]]

agent = Agent()

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """
    Returns a stream of JSON objects (NDJSON) representing progress updates.
    """
    q = queue.Queue()

    def progress_callback(step, status, details):
        q.put({
            "type": "progress",
            "step": step,
            "status": status,
            "details": details
        })

    def run_agent_thread():
        try:
            results = agent.run(request.ehr, request.lab_tests, progress_callback)
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
            q.put(None) # Sentinel to end stream

    # Start agent in a background thread
    thread = threading.Thread(target=run_agent_thread)
    thread.start()

    def event_generator():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return """
        <html>
            <body>
                <h1>Agent API is running</h1>
                <p>Use POST /api/analyze to submit data.</p>
            </body>
        </html>
        """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
