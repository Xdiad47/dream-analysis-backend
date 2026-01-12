from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Any, Dict
from datetime import datetime
import os
import json
import re

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

# ----------------------------
# NLTK setup (Render-safe)
# ----------------------------
NLTK_DATA_DIR = os.getenv("NLTK_DATA", "/opt/render/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

def ensure_nltk():
    # tokenizers
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)

    # stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

    # tagger (for POS tagging -> noun-based symbols)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DATA_DIR)

    # Some NLTK builds look for this newer name (safe to try)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", download_dir=NLTK_DATA_DIR)
        except Exception:
            pass

ensure_nltk()

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Dream Analysis AI API", version="2.0.0")

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
    temperature=0.4,     # lower = more consistent structured output
    max_tokens=900
)

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

from knowledge import psychology_knowledge

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=40
)

knowledge_chunks: List[str] = []
for doc in psychology_knowledge:
    chunks = text_splitter.split_text(doc.strip())
    knowledge_chunks.extend([c for c in chunks if c.strip()])

# keep memory + speed friendly
knowledge_chunks = knowledge_chunks[:120]

CHROMA_DIR = os.getenv("CHROMA_DIR", "/opt/render/chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

# Try loading persisted collection first (faster)
vectorstore = Chroma(
    collection_name="dream_psychology",
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# If empty, add your initial knowledge chunks
try:
    existing_count = vectorstore._collection.count()  # internal but works with Chroma
except Exception:
    existing_count = 0

if existing_count == 0:
    vectorstore.add_texts(texts=knowledge_chunks)
    try:
        vectorstore.persist()
    except Exception:
        pass

print("✅ AI components initialized")
print(f"📚 Knowledge chunks: {len(knowledge_chunks)}")

# ----------------------------
# Models (UI-friendly)
# ----------------------------
class DreamRequest(BaseModel):
    dreamText: str
    dreamDate: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    moodBeforeSleep: str = "Unknown"
    userId: Optional[str] = None

class SymbolInsight(BaseModel):
    symbol: str
    meaning: str
    evidence: str

class DreamResponse(BaseModel):
    # Existing fields (keep for backward compatibility)
    emotions: List[str]
    symbols: List[str]
    entities: List[Tuple[str, str]]
    analysisShort: str
    analysisFull: str
    sourcesUsed: int

    # New UI fields
    summary: str
    themes: List[str]
    symbolInsights: List[SymbolInsight]
    questions: List[str]
    actions: List[str]

# ----------------------------
# NLP helper (clean symbols/entities/emotions)
# ----------------------------
def extract_symbols_nouns(dream_text: str) -> List[str]:
    tokens = word_tokenize(dream_text)
    tagged = pos_tag(tokens)
    stop_words = set(stopwords.words("english"))

    nouns = []
    for w, tag in tagged:
        wl = w.lower().strip()
        if not wl.isalnum():
            continue
        if wl in stop_words:
            continue
        if len(wl) < 3:
            continue

        # Keep nouns only (NN, NNS, NNP, NNPS)
        if tag.startswith("NN"):
            nouns.append(wl)

    # de-dupe preserve order
    nouns = list(dict.fromkeys(nouns))
    return nouns[:12]

def extract_entities_proper_nouns(dream_text: str) -> List[Tuple[str, str]]:
    tokens = word_tokenize(dream_text)
    tagged = pos_tag(tokens)

    ents = []
    for w, tag in tagged:
        # Proper nouns only
        if tag in ("NNP", "NNPS"):
            if w.lower() in ("the", "a", "an", "i"):
                continue
            ents.append((w, "ENTITY"))

    # de-dupe preserve order
    ents = list(dict.fromkeys(ents))
    return ents[:10]

def detect_emotions(dream_text: str, mood_before_sleep: str) -> List[str]:
    emotion_keywords = {
        "fear": ["scared", "afraid", "terrified", "panic", "anxious", "worried", "nightmare"],
        "joy": ["happy", "excited", "joyful", "delighted", "pleased", "thrilled"],
        "calm": ["calm", "peaceful", "relaxed", "safe", "serene", "content", "quiet"],
        "freedom": ["free", "liberated", "unburdened", "light", "released"],
        "sadness": ["sad", "depressed", "lonely", "crying", "tears", "grief"],
        "anger": ["angry", "furious", "frustrated", "annoyed", "mad"],
        "confusion": ["confused", "lost", "uncertain", "puzzled"],
    }

    dream_lower = dream_text.lower()
    detected = set()

    for emotion, keywords in emotion_keywords.items():
        if any(k in dream_lower for k in keywords):
            detected.add(emotion)

    # Use mood as a hint (helps UX)
    mood = (mood_before_sleep or "").strip().lower()
    mood_map = {
        "calm": "calm",
        "peaceful": "calm",
        "relaxed": "calm",
        "happy": "joy",
        "anxious": "fear",
        "stress": "fear",
        "stressed": "fear",
        "sad": "sadness"
    }
    for key, val in mood_map.items():
        if key in mood:
            detected.add(val)

    return sorted(list(detected))

def extract_features(dream_text: str, mood_before_sleep: str) -> Dict[str, Any]:
    return {
        "symbols": extract_symbols_nouns(dream_text),
        "entities": extract_entities_proper_nouns(dream_text),
        "emotions": detect_emotions(dream_text, mood_before_sleep)
    }

# ----------------------------
# LLM prompt + safe JSON parsing
# ----------------------------
def build_ui_prompt(dream_text: str, context: str, features: Dict[str, Any], dream_date: str, mood_before_sleep: str) -> str:
    return f"""
You are a dream analysis assistant grounded in psychology. Use ONLY the PSYCHOLOGICAL KNOWLEDGE below.
Write in a warm, helpful tone.

PSYCHOLOGICAL KNOWLEDGE:
{context}

USER DREAM:
{dream_text}

META:
- Dream date: {dream_date}
- Mood before sleep: {mood_before_sleep}

EXTRACTED SIGNALS:
- Candidate symbols (nouns): {features["symbols"]}
- Entities: {features["entities"]}
- Detected emotions: {features["emotions"]}

Return STRICT JSON ONLY (no markdown, no code fences) in exactly this schema:

{{
  "summary": "1 sentence user-friendly summary",
  "themes": ["theme1","theme2","theme3","theme4"],
  "symbolInsights": [
    {{"symbol":"...","meaning":"...","evidence":"short quote from the dream"}},
    {{"symbol":"...","meaning":"...","evidence":"short quote from the dream"}},
    {{"symbol":"...","meaning":"...","evidence":"short quote from the dream"}}
  ],
  "questions": ["question1","question2"],
  "actions": ["action1","action2"]
}}

Rules:
- Themes: 3–6 items max.
- SymbolInsights: exactly 3 items if possible (use the dream’s strongest elements).
- Evidence: must be a short quote from the dream text (5–12 words).
- Actions must be practical and small (doable in 5–10 minutes).
- Do not mention astrology, zodiac, fortune, predictions, or mysticism.
""".strip()

def extract_json_object(text: str) -> Optional[dict]:
    # Remove code fences if any
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try to find first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def build_markdown_from_ui(ui: dict, dream_text: str, emotions: List[str], sources_used: int) -> str:
    sym_lines = []
    for s in ui.get("symbolInsights", []):
        sym_lines.append(f"- **{s.get('symbol','')}** — {s.get('meaning','')}  \n  _Evidence:_ “{s.get('evidence','')}”")

    q_lines = [f"- {q}" for q in ui.get("questions", [])]
    a_lines = [f"- {a}" for a in ui.get("actions", [])]

    theme_line = ", ".join(ui.get("themes", []))
    emo_line = ", ".join(emotions) if emotions else "—"

    return f"""**Dream Summary**
{ui.get("summary","")}

**Themes**
{theme_line if theme_line else "—"}

**Key Symbols**
{chr(10).join(sym_lines) if sym_lines else "—"}

**Reflection Questions**
{chr(10).join(q_lines) if q_lines else "—"}

**Action Steps**
{chr(10).join(a_lines) if a_lines else "—"}

**Detected Emotions**
{emo_line}

**Sources Used**
{sources_used}
""".strip()

def fallback_ui(features: Dict[str, Any], dream_text: str) -> dict:
    # Minimal safe fallback if JSON parsing fails
    top_syms = features["symbols"][:3]
    if not top_syms:
        top_syms = ["dream", "emotion", "context"]

    def quote_from_dream():
        words = dream_text.split()
        return " ".join(words[:10]) if len(words) >= 10 else dream_text

    return {
        "summary": "This dream reflects an emotional theme your mind is processing right now.",
        "themes": ["self-reflection", "emotional processing", "stress patterns"],
        "symbolInsights": [
            {"symbol": top_syms[0], "meaning": "A key element your mind is highlighting.", "evidence": quote_from_dream()},
            {"symbol": top_syms[1], "meaning": "Represents an emotional context or situation.", "evidence": quote_from_dream()},
            {"symbol": top_syms[2], "meaning": "Points to a personal concern or need.", "evidence": quote_from_dream()},
        ],
        "questions": [
            "What part of your current life feels most connected to this dream?",
            "What emotion stands out when you remember it?"
        ],
        "actions": [
            "Write 3 lines about what you think the dream is trying to process.",
            "Do a 5-minute calm routine (breathing, walk, or journaling)."
        ]
    }

# ----------------------------
# Core analysis
# ----------------------------
def analyze_dream_with_rag(dream_text: str, dream_date: str, mood_before_sleep: str) -> Dict[str, Any]:
    features = extract_features(dream_text, mood_before_sleep)

    # Retrieval
    relevant_docs = vectorstore.similarity_search(dream_text, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = build_ui_prompt(
        dream_text=dream_text,
        context=context,
        features=features,
        dream_date=dream_date,
        mood_before_sleep=mood_before_sleep
    )

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    ui = extract_json_object(raw)
    if not ui:
        ui = fallback_ui(features, dream_text)

    # Validate required keys exist
    ui.setdefault("summary", "A meaningful dream worth reflecting on.")
    ui.setdefault("themes", [])
    ui.setdefault("symbolInsights", [])
    ui.setdefault("questions", [])
    ui.setdefault("actions", [])

    # Backward-compatible fields
    analysis_short = ui["summary"].strip()
    sources_used = len(relevant_docs)
    analysis_full = build_markdown_from_ui(ui, dream_text, features["emotions"], sources_used)

    return {
        "ui": ui,
        "analysis_full": analysis_full,
        "analysis_short": analysis_short,
        "features": features,
        "sources_used": sources_used
    }

# ----------------------------
# API
# ----------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Dream Analysis AI API",
        "version": "2.0.0",
        "endpoints": {
            "analyze_v1": "/api/v1/dreams/analyze",
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

        result = analyze_dream_with_rag(
            dream_text=request.dreamText.strip(),
            dream_date=request.dreamDate,
            mood_before_sleep=request.moodBeforeSleep
        )

        ui = result["ui"]
        features = result["features"]

        # Ensure symbolInsights is shaped properly
        symbol_insights = []
        for s in ui.get("symbolInsights", [])[:3]:
            try:
                symbol_insights.append(SymbolInsight(**s))
            except Exception:
                pass

        return DreamResponse(
            emotions=features["emotions"],
            symbols=features["symbols"],
            entities=features["entities"],
            analysisShort=result["analysis_short"],
            analysisFull=result["analysis_full"],
            sourcesUsed=result["sources_used"],

            summary=ui.get("summary", result["analysis_short"]),
            themes=ui.get("themes", []),
            symbolInsights=symbol_insights,
            questions=ui.get("questions", []),
            actions=ui.get("actions", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
