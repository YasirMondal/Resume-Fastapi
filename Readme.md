🧠 Resume Information Extraction System

This project extracts structured information such as name, education, skills, experience, certifications, projects, and hobbies from resumes in PDF or DOCX format using Natural Language Processing (NLP).

The app supports both:

Local model inference (no API key required)

Hugging Face API-based inference (optional if you have a token)

🚀 Features

Upload and parse PDF/DOCX resumes

Named Entity Recognition (NER) using BERT-based model

Extracts key details into structured JSON

Optional integration with Supabase for file storage

Works completely offline when local models are downloaded

🏗 Tech Stack

Python 3.10+

Transformers, torch

FastAPI

PyMuPDF, docx2txt

Supabase-py (optional for upload)

📂 Project Structure

├── main.py                 # FastAPI app entry point
├── parser_utils.py         # Core parsing and model logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys etc.)
├── .gitignore
└── README.md

⚙ Installation & Setup

1. Clone the repository

git clone <your-repo-url>
cd <your-project-folder>

2. Create a virtual environment

python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

3. Install dependencies

pip install -r requirements.txt

🔑 Environment Variables

Create a file named .env in your project root and add:

HF_API_KEY=your_huggingface_token_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key

> The HF_API_KEY is only required if you use Hugging Face API calls.
The system also works fully offline when you use the local model.


🤖 Model Setup (Local Inference Mode)

To use the local model instead of API:

1. The model dslim/bert-base-NER will automatically download and cache inside:

~/.cache/huggingface/hub/


2. Once downloaded, it runs locally — no internet or API token needed.


3. The project automatically uses the local pipeline for inference if configured.


▶ Run the Application

python main.py

The FastAPI server will start at:

http://127.0.0.1:8000

You can then upload a resume and view structured results in JSON format.

🧩 Example Output

{
  "name": "John Doe",
  "education": {"entries": ["B.Tech in Computer Science, IIT Delhi"]},
  "experience": {"summary_lines": ["Software Engineer at ABC Corp (2021-2024)"]},
  "skills": ["python", "flask", "pandas", "sql"],
  "certifications": ["AWS Certified Developer"],
  "projects": ["AI Resume Parser - github.com/johndoe/ai-parser"],
  "hobbies": ["reading", "music"],
  "introduction": "Passionate developer with strong skills in Python and Data Engineering."
}