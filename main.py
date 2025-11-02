# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from supabase import create_client
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import shutil
import uuid

# Local parser utils (now using transformers locally when possible)
from parser_utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    call_local_ner,
    build_structured_fields,
    call_local_qa,
    try_load_models
)

load_dotenv()
# Try to ensure models are loaded on startup (this will trigger download if needed)
try_load_models()

# Env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "resumes")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "resume_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "candidates")

# Clients
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase URL/Key missing in env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing in env")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DBNAME]
candidates_col = db[MONGO_COLLECTION]

app = FastAPI(title="Resume Upload + Extractor (local models)")

TMP_DIR = "/tmp/resumes"
os.makedirs(TMP_DIR, exist_ok=True)

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    # Validate extension
    filename = file.filename
    if not filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only .pdf and .docx allowed")

    # Save temporary file
    uid = str(uuid.uuid4())
    tmp_path = os.path.join(TMP_DIR, f"{uid}_{filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text
    try:
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(tmp_path)
        else:
            text = extract_text_from_docx(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    # Call local NER (may return [] if models not available)
    ner_text = text[:45000]  # keep under limits if chunking is needed
    try:
        ner_resp = call_local_ner(ner_text)
    except Exception as e:
        ner_resp = []

    # Build structured fields (uses heuristics + NER output)
    structured_data = build_structured_fields(text, ner_resp)

    # Upload file to Supabase storage
    storage_path = f"{uid}/{filename}"
    try:
        with open(tmp_path, "rb") as f:
            res = supabase.storage.from_(SUPABASE_BUCKET).upload(storage_path, f)
            # supabase python client returns a dict; check for error key
            if isinstance(res, dict) and res.get("error"):
                raise Exception(res["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase upload error: {e}")

    # Insert metadata into Supabase table "resumes_metadata"
    metadata = {
        "file_name": filename,
        "storage_path": storage_path,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    try:
        insert_res = supabase.table("resumes_metadata").insert(metadata).execute()
        if isinstance(insert_res, dict) and insert_res.get("error"):
            raise Exception(insert_res["error"])
        supa_data = None
        try:
            supa_data = insert_res.get("data")
        except Exception:
            supa_data = insert_res  # fallback
        metadata_id = None
        if isinstance(supa_data, list) and len(supa_data) > 0:
            metadata_id = supa_data[0].get("id") or supa_data[0].get("metadata_id")
        elif isinstance(supa_data, dict) and supa_data.get("id"):
            metadata_id = supa_data.get("id")
        else:
            metadata_id = str(uuid.uuid4())
    except Exception:
        # proceed with generated id to avoid blocking entire flow
        metadata_id = str(uuid.uuid4())

    # Insert structured candidate into MongoDB
    candidate_doc = {
        "candidate_id": metadata_id,
        "file_name": filename,
        "storage_path": storage_path,
        "uploaded_at": datetime.utcnow(),
        "raw_text_snippet": text[:3000],
        **structured_data
    }
    try:
        candidates_col.insert_one(candidate_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB insert failed: {e}")

    return JSONResponse(status_code=201, content={"message": "Uploaded and processed", "candidate_id": metadata_id})

@app.get("/candidates")
def list_candidates(limit: int = 50):
    docs = candidates_col.find({}, {"_id": 0}).limit(limit)
    results = []
    for d in docs:
        summary = {
            "candidate_id": d.get("candidate_id"),
            "name": d.get("name",""),
            "skills": d.get("skills", []),
            "uploaded_at": d.get("uploaded_at"),
            "brief_intro": d.get("introduction","")
        }
        results.append(summary)
    return results

@app.get("/candidate/{candidate_id}")
def get_candidate(candidate_id: str):
    d = candidates_col.find_one({"candidate_id": candidate_id}, {"_id": 0})
    if not d:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return d

@app.post("/ask/{candidate_id}")
def ask_candidate(candidate_id: str, question: dict):
    """
    Expects JSON body: {"question": "When did X graduate?"}
    """
    q = question.get("question")
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'question' in body")

    d = candidates_col.find_one({"candidate_id": candidate_id}, {"_id": 0})
    if not d:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Build a context string from the candidate doc
    context_parts = []
    for key in ["introduction", "education", "experience", "skills", "certifications", "projects", "hobbies"]:
        val = d.get(key)
        if val:
            # Convert dicts/lists to readable text
            if isinstance(val, dict):
                part = f"{key.upper()}:\n"
                for k, v in val.items():
                    part += f"{k}: {v}\n"
            elif isinstance(val, list):
                part = f"{key.upper()}:\n" + "\n".join([str(x) for x in val]) + "\n"
            else:
                part = f"{key.upper()}:\n{val}\n"
            context_parts.append(part)
    context = "\n".join(context_parts)

    try:
        hf_resp = call_local_qa(q, context)
        # Normalize responses: pipeline returns dict or list
        if isinstance(hf_resp, dict) and hf_resp.get("answer"):
            answer = hf_resp.get("answer")
        elif isinstance(hf_resp, list) and len(hf_resp) > 0:
            # list of answers
            if isinstance(hf_resp[0], dict) and hf_resp[0].get("answer"):
                answer = hf_resp[0].get("answer")
            else:
                answer = str(hf_resp[0])
        elif isinstance(hf_resp, dict) and hf_resp.get("error"):
            raise Exception(hf_resp.get("error"))
        else:
            answer = str(hf_resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {e}")

    return {"candidate_id": candidate_id, "question": q, "answer": answer}