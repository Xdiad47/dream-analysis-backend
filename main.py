from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ✅ Lightweight embeddings (NO torch / NO CUDA)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import os

# ----------------------------
# NLTK setup (Render-safe)
# ----------------------------
# Render containers are ephemeral, but /opt/render is writable.
NLTK_DATA_DIR = os.getenv("NLTK_DATA", "/opt/render/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

ensure_nltk()

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Dream Analysis AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# ENV
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable")

# ----------------------------
# AI components (startup)
# ----------------------------
print("🚀 Initializing AI components...")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=1024
)

# ✅ No torch. Uses FastEmbed (CPU ONNX)
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"  # solid general embedding model
)

# Load psychology knowledge
from knowledge import psychology_knowledge

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

knowledge_chunks = []
# Keep your memory limits
for doc in psychology_knowledge[:20]:
    chunks = text_splitter.split_text(doc.strip())
    knowledge_chunks.extend(chunks)

knowledge_chunks = knowledge_chunks[:50]

# ✅ Persist Chroma to disk so it survives restarts inside same container lifetime
CHROMA_DIR = os.getenv("CHROMA_DIR", "/opt/render/chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

vectorstore = Chroma.from_texts(
    texts=knowledge_chunks,
    embedding=embeddings,
    collection_name="dream_psychology",
    persist_directory=CHROMA_DIR
)

print("✅ AI components initialized")

# ----------------------------
# Models
# ----------------------------
class DreamRequest(BaseModel):
    dreamText: str
    dreamDate: str = datetime.now().strftime("%Y-%m-%d")
    moodBeforeSleep: str = "Unknown"
    userId: str | None = None

class DreamResponse(BaseModel):
    emotions: list
    symbols: list
    entities: list
    analysisShort: str
    analysisFull: str
    sourcesUsed: int

# ----------------------------
# NLP helper
# ----------------------------
def extract_entities_and_emotions(dream_text: str):
    tokens = word_tokenize(dream_text.lower())
    stop_words = set(stopwords.words("english"))

    symbols = [w for w in tokens if w.isalnum() and len(w) > 3 and w not in stop_words]
    entities = [(w, "ENTITY") for w in dream_text.split() if w[:1].isupper() and len(w) > 1]

    emotion_keywords = {
        "fear": ["scared", "afraid", "terrified", "panic", "anxious", "worried"],
        "joy": ["happy", "excited", "joyful", "delighted", "pleased"],
        "sadness": ["sad", "depressed", "lonely", "crying", "tears"],
        "anger": ["angry", "furious", "frustrated", "annoyed", "mad"],
        "confusion": ["confused", "lost", "uncertain", "puzzled"],
    }

    dream_lower = dream_text.lower()
    detected_emotions = []
    for emotion, keywords in emotion_keywords.items():
        if any(k in dream_lower for k in keywords):
            detected_emotions.append(emotion)

    return {
        "entities": entities[:10],
        "symbols": list(set(symbols))[:10],
        "emotions": detected_emotions
    }

def create_dream_prompt(dream_text, context, entities, symbols, emotions):
    return f"""You are an AI dream analyst trained in psychology and neuroscience. Analyze the dream using ONLY the provided psychological knowledge.

PSYCHOLOGICAL KNOWLEDGE:
{context}

DREAM TO ANALYZE:
{dream_text}

EXTRACTED SIGNALS:
- Entities: {entities}
- Symbols: {symbols}
- Detected Emotions: {emotions}

Provide a personalized dream analysis covering:
1. Primary Meaning
2. Emotional Context
3. Psychological Insight
4. Actionable Reflection

Be compassionate, evidence-based, and actionable. Avoid mysticism.

ANALYSIS:
"""

def analyze_dream_with_rag(dream_text: str):
    nlp_features = extract_entities_and_emotions(dream_text)

    # Retrieval
    relevant_docs = vectorstore.similarity_search(dream_text, k=2)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = create_dream_prompt(
        dream_text=dream_text,
        context=context,
        entities=nlp_features["entities"],
        symbols=nlp_features["symbols"],
        emotions=nlp_features["emotions"]
    )

    response = llm.invoke(prompt)
    analysis_full = response.content

    # Simple short summary
    analysis_short = analysis_full.strip()
    if len(analysis_short) > 220:
        analysis_short = analysis_short[:220].rsplit(" ", 1)[0] + "..."

    return {
        "analysis_full": analysis_full,
        "analysis_short": analysis_short,
        "nlp_features": nlp_features,
        "sources_used": len(relevant_docs)
    }

# ----------------------------
# API
# ----------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Dream Analysis AI API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/v1/dreams/analyze",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ai_status": "loaded",
        "vector_db": "ready",
        "knowledge_chunks": len(knowledge_chunks),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/dreams/analyze", response_model=DreamResponse)
def analyze_dream(request: DreamRequest):
    try:
        if not request.dreamText or len(request.dreamText.strip()) < 10:
            raise HTTPException(status_code=400, detail="Dream text too short (minimum 10 characters)")

        result = analyze_dream_with_rag(request.dreamText)

        return DreamResponse(
            emotions=result["nlp_features"]["emotions"],
            symbols=result["nlp_features"]["symbols"],
            entities=result["nlp_features"]["entities"],
            analysisShort=result["analysis_short"],
            analysisFull=result["analysis_full"],
            sourcesUsed=result["sources_used"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

