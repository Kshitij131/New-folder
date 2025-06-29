# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
import shutil
import uuid
from backend_core import DataManager

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Save uploaded file
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{upload_file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

# Endpoint to accept files from frontend
@app.post("/upload")
async def upload_files(
    clients: UploadFile = File(...),
    workers: UploadFile = File(...),
    tasks: UploadFile = File(...)
):
    try:
        clients_path = save_upload_file(clients)
        workers_path = save_upload_file(workers)
        tasks_path = save_upload_file(tasks)

        dm = DataManager()
        load_status = dm.load_data_from_files(clients_path, workers_path, tasks_path)

        if load_status["status"] == "error":
            return JSONResponse(status_code=400, content=load_status)

        validation_errors = dm.run_all_validations()
        return {"status": "success", "errors": [e.__dict__ for e in validation_errors]}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Natural language search
@app.post("/nl_search")
async def nl_search(query: str = Form(...)):
    try:
        dm = DataManager()
        results = dm.natural_language_search(query)
        return {"status": "success", "results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Natural language modify
@app.post("/nl_modify")
async def nl_modify(command: str = Form(...)):
    try:
        dm = DataManager()
        updated_data = dm.natural_language_modify(command)
        return {"status": "success", "updated": updated_data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Row correction suggestions
@app.get("/suggest_corrections")
async def suggest_corrections():
    try:
        dm = DataManager()
        suggestions = dm.suggest_data_corrections()
        return {"status": "success", "suggestions": suggestions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# AI Rule Recommendations
@app.get("/ai_rule_recommendations")
async def ai_rule_recommendations():
    try:
        dm = DataManager()
        rule_suggestions = dm.get_ai_rule_recommendations()
        return {"status": "success", "rules": rule_suggestions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
