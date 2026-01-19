# 🧠 Dream Analysis AI - Backend

Python backend API for the Dream Analysis AI application. It provides AI-powered dream interpretation using NLP, LangChain, and RAG (Retrieval-Augmented Generation) with a psychology-informed knowledge base.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat)

## 🚀 Features

- AI-powered dream interpretation API using LLMs and RAG.
- NLP preprocessing of dream text (tokenization, cleaning, emotion/theme extraction).
- Knowledge-based reasoning using curated dream psychology data.
- Optimized startup to reduce cold-start time on small Render instances.
- Ready-to-deploy configuration for Render and Railway.

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Framework**: FastAPI
- **Server**: Uvicorn
- **AI Orchestration**: LangChain
- **NLP**: NLTK and related libraries
- **Vector / Knowledge**: In-memory or file-based knowledge, extendable to vector DB
- **Deployment**: Render / Railway using `Procfile` and `railway.json`

## 📁 Project Structure

```text
.
├── main.py          # FastAPI app and endpoints
├── knowledge.py     # Dream interpretation knowledge and prompt helpers
├── requirements.txt # Python dependencies
├── Procfile         # Render start command
├── railway.json     # Railway deployment configuration
└── .python-version  # Python version pin for hosting
```

## 🔧 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Xdiad47/Dream-Analysis.git
cd Dream-Analysis
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables

Create a `.env` file in the project root (this file must stay **local** and be git-ignored):

```env
GROQ_API_KEY=your_groq_api_key_here
# Add any other secrets here (DB URLs, etc.)
```

Make sure `.env` is listed in `.gitignore` so it never gets committed.

## ▶️ Run the API Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API base URL: `http://localhost:8000`
- Interactive docs (Swagger): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 Example Endpoints

### Health check

```http
GET /
```

**Response:**

```json
{
  "status": "Dream Analysis API is running"
}
```

### Analyze dream

```http
POST /analyze
Content-Type: application/json

{
  "dream_text": "I was flying over a dark city and then falling.",
  "user_id": "optional-user-id"
}
```

**Response (example):**

```json
{
  "interpretation": "This dream may reflect a desire for freedom mixed with anxiety about losing control.",
  "themes": ["freedom", "control", "anxiety"],
  "emotions": ["fear", "excitement"]
}
```

The actual response depends on your prompt design, knowledge base, and LLM configuration.

## ☁️ Deployment

### Render

1. Create a new **Web Service** from this repo.
2. Set runtime to **Python 3.11** (matches `.python-version`).
3. Build command (if needed):

```bash
pip install -r requirements.txt
```

4. Start command (from `Procfile`):

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

5. Add environment variable in Render dashboard:

- `GROQ_API_KEY = your_groq_api_key_here`

### Railway

The `railway.json` file is prepared for deployment. Typical commands:

```bash
railway up
```

Configure environment variables in the Railway dashboard (same as for Render).

## 🔐 Security & Secrets

- Never commit your `GROQ_API_KEY` or any other secret.
- Keep secrets only in `.env` or the hosting provider's environment variable settings.
- Ensure `.env`, any `*_secrets.py`, and config files with keys are ignored by git.

## 🧪 Testing

If you add tests (recommended):

```bash
pip install pytest httpx
pytest
```

You can also use tools like Postman or curl to manually test the API endpoints.

## 🛤️ Roadmap

- Improve dream theme and emotion classification.
- Add user-specific memory and recurring-pattern analysis.
- Optional: connect to mobile client (Flutter) as a public API.
- Optional: plug in a vector database for richer RAG retrieval.

## 👨‍💻 Author

**Diadem Nath**

- GitHub: [@Xdiad47](https://github.com/Xdiad47)
- LinkedIn: [Diadem Nath](https://www.linkedin.com/in/diadem-nath-a5396152/)
- Email: [mail2diadem@gmail.com](mailto:mail2diadem@gmail.com)
- Location: Guwahati, India

## 🔗 Related Projects

- [Dream Analysis - Full Stack](https://github.com/Xdiad47/Dream-Analysis) - Complete project with Flutter mobile app and Python backend

---

⭐ Star this repo if you find it helpful!
