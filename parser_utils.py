# parser_utils.py
import os
import docx2txt
import fitz  # PyMuPDF
import re
from dateutil import parser as dateparser

# Local transformers (will auto-download the models on first run)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Model names (local use)
NER_MODEL = "dslim/bert-base-NER"
QA_MODEL = "deepset/roberta-base-squad2"

# Pipelines (initialized below if possible)
ner_pipeline = None
qa_pipeline = None
_models_loaded = False

def try_load_models():
    """
    Attempt to load transformers pipelines once. If it fails,
    we keep TRANSFORMERS_AVAILABLE = False behavior and rely on heuristics.
    """
    global ner_pipeline, qa_pipeline, _models_loaded
    if _models_loaded:
        return
    if not TRANSFORMERS_AVAILABLE:
        _models_loaded = False
        return
    try:
        # NER pipeline with simple aggregation
        ner_pipeline = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
        # QA pipeline
        qa_pipeline = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL)
        _models_loaded = True
    except Exception:
        # If any model fails to load, keep flags false and let heuristics work
        ner_pipeline = None
        qa_pipeline = None
        _models_loaded = False

# Try load on import (harmless; will auto-download)
try_load_models()

def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)

def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path)

def call_local_ner(text: str):
    """
    Returns list of entities similar to HF inference output when possible.
    If transformers pipelines aren't available, return an empty list for downstream heuristics.
    """
    if _models_loaded and ner_pipeline:
        # transformers' pipeline may have token limits; chunk if necessary
        max_chunk = 45000
        text_to_use = text[:max_chunk]
        try:
            ents = ner_pipeline(text_to_use)
            # Normalize into list of dicts with keys 'word' and 'entity_group'
            normalized = []
            for e in ents:
                word = e.get("word") if "word" in e else e.get("entity")
                normalized.append({
                    "word": word,
                    "entity_group": e.get("entity_group") or e.get("entity")
                })
            return normalized
        except Exception:
            return []
    else:
        return []

def call_local_qa(question: str, context: str, top_k: int = 1):
    """
    Run local QA pipeline. Return a dict/list in a format similar to HF inference.
    If model not available, return a fallback answer using heuristics (simple search).
    """
    if _models_loaded and qa_pipeline:
        # QA models prefer contexts under a certain length; truncate if necessary
        max_context = 20000
        ctx = context if len(context) <= max_context else context[-max_context:]  # take last part
        try:
            res = qa_pipeline(question=question, context=ctx, top_k=top_k)
            return res
        except Exception as e:
            return {"error": str(e)}
    # Fallback simple heuristic: search for lines that contain keywords from question
    q = question.lower()
    candidates = []
    for line in context.splitlines():
        if len(line.strip()) < 3:
            continue
        if any(word in line.lower() for word in q.split()[:3]):  # naive
            candidates.append(line.strip())
    if candidates:
        return {"answer": candidates[0]}
    return {"answer": "No clear answer found in stored data."}

# Simple heuristics to convert text + NER output into the requested schema
COMMON_SKILLS = [
    "python","java","c++","c","javascript","sql","nosql","mongodb","postgresql","mysql",
    "pandas","numpy","scikit-learn","tensorflow","keras","pytorch","fastapi","flask",
    "docker","kubernetes","aws","gcp","azure","html","css","react","node","spark","hadoop"
]

DEGREE_KEYWORDS = [
    "bachelor", "bsc", "b.tech", "btech", "bs", "master", "msc", "m.tech", "mtech", "ms", "phd", "diploma"
]

def build_structured_fields(text: str, ner_entities: list):
    """
    Outputs:
    {
      "name": "",
      "education": {},
      "experience": {},
      "skills": [],
      "hobbies": [],
      "certifications": [],
      "projects": [],
      "introduction": ""
    }
    """
    lower_text = text.lower()

    # Name detection: prefer first PER entity
    name = None
    for ent in (ner_entities or []):
        if ent.get("entity_group") and ent.get("entity_group").upper().startswith("PER"):
            name = ent.get("word")
            break

    # Skills: find common keywords
    found_skills = set()
    for skill in COMMON_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", lower_text):
            found_skills.add(skill)

    # Education heuristics: lines with degree keywords or university names
    education = {}
    edu_lines = []
    for line in text.splitlines():
        if any(k in line.lower() for k in DEGREE_KEYWORDS) or re.search(r"\b(university|college|institute|school)\b", line.lower()):
            if len(line.strip()) > 5:
                edu_lines.append(line.strip())

    if edu_lines:
        education["entries"] = edu_lines
    else:
        # fallback: ORG entities
        orgs = [e["word"] for e in (ner_entities or []) if e.get("entity_group") and e.get("entity_group").upper().startswith("ORG")]
        if orgs:
            education["entries"] = orgs[:3]

    # Experience heuristics: lines with years or role keywords
    experiences = []
    year_pattern = re.compile(r"(19|20)\d{2}")
    for line in text.splitlines():
        l = line.strip()
        if len(l) == 0:
            continue
        if year_pattern.search(l) and any(k in l.lower() for k in ("intern", "engineer", "manager", "associate", "analyst", "developer")):
            experiences.append(l)
    # Add ORG mentions as short entries
    orgs = [e["word"] for e in (ner_entities or []) if e.get("entity_group") and e.get("entity_group").upper().startswith("ORG")]
    for o in orgs[:5]:
        experiences.append(f"Worked at {o}")

    # Certifications & projects
    certs = []
    projects = []
    for line in text.splitlines():
        low = line.lower()
        if "certif" in low or "course" in low or "certificate" in low:
            certs.append(line.strip())
        if "project" in low or "github.com" in low:
            projects.append(line.strip())

    # Hobbies
    hobby_keywords = ["reading", "travelling", "travel", "music", "photography", "gaming", "sports", "cricket", "football"]
    hobbies = []
    for h in hobby_keywords:
        if re.search(r"\b" + re.escape(h) + r"\b", lower_text):
            hobbies.append(h)

    # Intro: first non-empty line with some length
    intro = ""
    for line in text.splitlines():
        if len(line.strip()) > 30:
            intro = line.strip()
            break

    structured = {
        "name": name or "",
        "education": education or {},
        "experience": {"summary_lines": experiences} if experiences else {},
        "skills": sorted(list(found_skills)),
        "hobbies": hobbies,
        "certifications": certs,
        "projects": projects,
        "introduction": intro
    }
    return structured