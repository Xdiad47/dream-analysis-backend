from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import os

# Download NLTK data on first run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize FastAPI
app = FastAPI(title="Dream Analysis AI API", version="1.0.0")

# CORS - Allow Android app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Initialize AI components (load once at startup)
print("🚀 Initializing AI components...")

# ============================================
# MODEL 1: GROQ LLM (for dream analysis generation)
# This uses Groq's API - llama-3.1-8b-instant model
# ============================================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # This is a Groq model
    temperature=0.7,
    max_tokens=1024
)

# ============================================
# MODEL 2: HUGGINGFACE EMBEDDINGS (for semantic search)
# This downloads from HuggingFace Hub automatically
# It converts text to vectors for similarity search
# ============================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # HuggingFace model (auto-downloads)
    model_kwargs={'device': 'cpu'},  # Use CPU (not GPU)
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 1,  # Process one at a time to save memory
        'show_progress_bar': False
    }
)

# Load psychology knowledge (from knowledge.py)
from knowledge import psychology_knowledge

# Create vector store with MEMORY OPTIMIZATION
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Reduced from 500 to save memory
    chunk_overlap=30  # Reduced from 50
)

knowledge_chunks = []
for doc in psychology_knowledge[:20]:  # Limit to first 20 docs
    chunks = text_splitter.split_text(doc.strip())
    knowledge_chunks.extend(chunks)

# Limit total chunks to reduce memory usage
knowledge_chunks = knowledge_chunks[:50]  # Max 50 chunks

vectorstore = Chroma.from_texts(
    texts=knowledge_chunks,
    embedding=embeddings,  # Uses HuggingFace embeddings
    collection_name="dream_psychology"
)

print("✅ AI components initialized")

# ============================================
# MODELS (Request/Response)
# ============================================

class DreamRequest(BaseModel):
    dreamText: str
    dreamDate: str = datetime.now().strftime("%Y-%m-%d")
    moodBeforeSleep: str = "Unknown"
    userId: str = None

class DreamResponse(BaseModel):
    emotions: list
    symbols: list
    entities: list
    analysisShort: str
    analysisFull: str
    sourcesUsed: int

# ============================================
# HELPER FUNCTIONS (From Jupyter)
# ============================================

def extract_entities_and_emotions(dream_text):
    """Extract entities, emotions, and key themes from dream text using NLTK"""
    # Tokenize
    tokens = word_tokenize(dream_text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Extract symbols (nouns/verbs longer than 3 chars)
    symbols = [word for word in tokens if word.isalnum() and len(word) > 3 and word not in stop_words]
    
    # Simple entity extraction (capitalized words in original text)
    entities = [(word, 'ENTITY') for word in dream_text.split() if word[0].isupper() and len(word) > 1]
    
    emotion_keywords = {
        'fear': ['scared', 'afraid', 'terrified', 'panic', 'anxious', 'worried'],
        'joy': ['happy', 'excited', 'joyful', 'delighted', 'pleased'],
        'sadness': ['sad', 'depressed', 'lonely', 'crying', 'tears'],
        'anger': ['angry', 'furious', 'frustrated', 'annoyed', 'mad'],
        'confusion': ['confused', 'lost', 'uncertain', 'puzzled']
    }
    
    detected_emotions = []
    dream_lower = dream_text.lower()
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in dream_lower for keyword in keywords):
            detected_emotions.append(emotion)
    
    return {
        'entities': entities[:10],  # Limit to 10
        'symbols': list(set(symbols))[:10],
        'emotions': detected_emotions
    }

def create_dream_prompt(dream_text, context, entities, symbols, emotions):
    """Create a structured prompt for dream analysis"""
    return f"""You are an AI dream analyst trained in psychology and neuroscience. Analyze the following dream using the provided psychological knowledge.

PSYCHOLOGICAL KNOWLEDGE:
{context}

DREAM TO ANALYZE:
{dream_text}

EXTRACTED SIGNALS:
- Entities: {entities}
- Symbols: {symbols}
- Detected Emotions: {emotions}

Provide a personalized dream analysis covering:
1. **Primary Meaning**: What this dream likely represents in the dreamer's life
2. **Emotional Context**: The emotional state reflected in the dream
3. **Psychological Insight**: Underlying patterns or unresolved issues
4. **Actionable Reflection**: What the dreamer can reflect on or address in waking life

Keep the analysis compassionate, evidence-based, and actionable. Avoid mysticism. Focus on psychological and emotional insights.

ANALYSIS:"""

def analyze_dream_with_rag(dream_text, vectorstore, llm):
    """Analyze dream using RAG pipeline"""
    # Extract NLP features
    nlp_features = extract_entities_and_emotions(dream_text)
    
    # Retrieve relevant knowledge (reduced to k=2 to save memory)
    relevant_docs = vectorstore.similarity_search(dream_text, k=2)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Build prompt
    prompt = create_dream_prompt(
        dream_text=dream_text,
        context=context,
        entities=nlp_features['entities'],
        symbols=nlp_features['symbols'],
        emotions=nlp_features['emotions']
    )
    
    # Get response from LLM (uses Groq API)
    response = llm.invoke(prompt)
    
    # Extract short summary
    analysis_full = response.content
    analysis_short = ""
    
    if "Primary Meaning" in analysis_full:
        start = analysis_full.find("Primary Meaning")
        end = analysis_full.find("### 2.") if "### 2." in analysis_full else len(analysis_full)
        primary = analysis_full[start:end].replace("### 1. **Primary Meaning**", "").strip()
        primary = primary.replace("**", "").replace("###", "").strip()
        sentences = primary.split('. ')
        analysis_short = '. '.join(sentences[:2]) + '.'
    
    return {
        'analysis_full': analysis_full,
        'analysis_short': analysis_short,
        'nlp_features': nlp_features,
        'sources_used': len(relevant_docs)
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Health check endpoint"""
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
    """Detailed health check"""
    return {
        "status": "healthy",
        "ai_status": "loaded",
        "vector_db": "ready",
        "knowledge_chunks": len(knowledge_chunks),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/dreams/analyze", response_model=DreamResponse)
def analyze_dream(request: DreamRequest):
    """
    Analyze a dream and return psychological insights
    """
    try:
        # Validate input
        if not request.dreamText or len(request.dreamText) < 10:
            raise HTTPException(status_code=400, detail="Dream text too short (minimum 10 characters)")
        
        # Analyze dream
        result = analyze_dream_with_rag(request.dreamText, vectorstore, llm)
        
        # Return response
        return DreamResponse(
            emotions=result['nlp_features']['emotions'],
            symbols=result['nlp_features']['symbols'],
            entities=result['nlp_features']['entities'],
            analysisShort=result['analysis_short'],
            analysisFull=result['analysis_full'],
            sourcesUsed=result['sources_used']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000































#from fastapi import FastAPI, HTTPException
#from fastapi.middleware.cors import CORSMiddleware
#from pydantic import BaseModel
#from groq import Groq
#from langchain_groq import ChatGroq
#from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
#import chromadb
#import nltk
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from datetime import datetime
#import os
#
## Download NLTK data on first run
#try:
#    nltk.data.find('tokenizers/punkt')
#except LookupError:
#    nltk.download('punkt')
#try:
#    nltk.data.find('corpora/stopwords')
#except LookupError:
#    nltk.download('stopwords')
#
## Initialize FastAPI
#app = FastAPI(title="Dream Analysis AI API", version="1.0.0")
#
## CORS - Allow Android app to call this API
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # In production, specify your Android app domain
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)
#
## Load environment variables
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#
#
## Initialize AI components (load once at startup)
#print("🚀 Initializing AI components...")
#
## Initialize Groq
#llm = ChatGroq(
#    groq_api_key=GROQ_API_KEY,
#    model_name="llama-3.1-8b-instant",
#    temperature=0.7,
#    max_tokens=1024
#)
#
## Initialize embeddings
#embeddings = HuggingFaceEmbeddings(
#    model_name="sentence-transformers/all-MiniLM-L6-v2",
#    model_kwargs={'device': 'cpu'}
#)
#
## Load psychology knowledge (from knowledge.py)
#from knowledge import psychology_knowledge
#
## Create vector store
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#knowledge_chunks = []
#for doc in psychology_knowledge:
#    chunks = text_splitter.split_text(doc.strip())
#    knowledge_chunks.extend(chunks)
#
#vectorstore = Chroma.from_texts(
#    texts=knowledge_chunks,
#    embedding=embeddings,
#    collection_name="dream_psychology"
#)
#
#print("✅ AI components initialized")
#
## ============================================
## MODELS (Request/Response)
## ============================================
#
#class DreamRequest(BaseModel):
#    dreamText: str
#    dreamDate: str = datetime.now().strftime("%Y-%m-%d")
#    moodBeforeSleep: str = "Unknown"
#    userId: str = None
#
#class DreamResponse(BaseModel):
#    emotions: list
#    symbols: list
#    entities: list
#    analysisShort: str
#    analysisFull: str
#    sourcesUsed: int
#
## ============================================
## HELPER FUNCTIONS (From Jupyter)
## ============================================
#
#def extract_entities_and_emotions(dream_text):
#    """Extract entities, emotions, and key themes from dream text using NLTK"""
#    # Tokenize
#    tokens = word_tokenize(dream_text.lower())
#    stop_words = set(stopwords.words('english'))
#    
#    # Extract symbols (nouns/verbs longer than 3 chars)
#    symbols = [word for word in tokens if word.isalnum() and len(word) > 3 and word not in stop_words]
#    
#    # Simple entity extraction (capitalized words in original text)
#    entities = [(word, 'ENTITY') for word in dream_text.split() if word[0].isupper() and len(word) > 1]
#    
#    emotion_keywords = {
#        'fear': ['scared', 'afraid', 'terrified', 'panic', 'anxious', 'worried'],
#        'joy': ['happy', 'excited', 'joyful', 'delighted', 'pleased'],
#        'sadness': ['sad', 'depressed', 'lonely', 'crying', 'tears'],
#        'anger': ['angry', 'furious', 'frustrated', 'annoyed', 'mad'],
#        'confusion': ['confused', 'lost', 'uncertain', 'puzzled']
#    }
#    
#    detected_emotions = []
#    dream_lower = dream_text.lower()
#    for emotion, keywords in emotion_keywords.items():
#        if any(keyword in dream_lower for keyword in keywords):
#            detected_emotions.append(emotion)
#    
#    return {
#        'entities': entities[:10],  # Limit to 10
#        'symbols': list(set(symbols))[:10],
#        'emotions': detected_emotions
#    }
#
#def create_dream_prompt(dream_text, context, entities, symbols, emotions):
#    """Create a structured prompt for dream analysis"""
#    return f"""You are an AI dream analyst trained in psychology and neuroscience. Analyze the following dream using the provided psychological knowledge.
#
#PSYCHOLOGICAL KNOWLEDGE:
#{context}
#
#DREAM TO ANALYZE:
#{dream_text}
#
#EXTRACTED SIGNALS:
#- Entities: {entities}
#- Symbols: {symbols}
#- Detected Emotions: {emotions}
#
#Provide a personalized dream analysis covering:
#1. **Primary Meaning**: What this dream likely represents in the dreamer's life
#2. **Emotional Context**: The emotional state reflected in the dream
#3. **Psychological Insight**: Underlying patterns or unresolved issues
#4. **Actionable Reflection**: What the dreamer can reflect on or address in waking life
#
#Keep the analysis compassionate, evidence-based, and actionable. Avoid mysticism. Focus on psychological and emotional insights.
#
#ANALYSIS:"""
#
#def analyze_dream_with_rag(dream_text, vectorstore, llm):
#    """Analyze dream using RAG pipeline"""
#    # Extract NLP features
#    nlp_features = extract_entities_and_emotions(dream_text)
#    
#    # Retrieve relevant knowledge
#    relevant_docs = vectorstore.similarity_search(dream_text, k=4)
#    context = "\n\n".join([doc.page_content for doc in relevant_docs])
#    
#    # Build prompt
#    prompt = create_dream_prompt(
#        dream_text=dream_text,
#        context=context,
#        entities=nlp_features['entities'],
#        symbols=nlp_features['symbols'],
#        emotions=nlp_features['emotions']
#    )
#    
#    # Get response from LLM
#    response = llm.invoke(prompt)
#    
#    # Extract short summary
#    analysis_full = response.content
#    analysis_short = ""
#    
#    if "Primary Meaning" in analysis_full:
#        start = analysis_full.find("Primary Meaning")
#        end = analysis_full.find("### 2.") if "### 2." in analysis_full else len(analysis_full)
#        primary = analysis_full[start:end].replace("### 1. **Primary Meaning**", "").strip()
#        primary = primary.replace("**", "").replace("###", "").strip()
#        sentences = primary.split('. ')
#        analysis_short = '. '.join(sentences[:2]) + '.'
#    
#    return {
#        'analysis_full': analysis_full,
#        'analysis_short': analysis_short,
#        'nlp_features': nlp_features,
#        'sources_used': len(relevant_docs)
#    }
#
## ============================================
## API ENDPOINTS
## ============================================
#
#@app.get("/")
#def root():
#    """Health check endpoint"""
#    return {
#        "status": "online",
#        "message": "Dream Analysis AI API",
#        "version": "1.0.0",
#        "endpoints": {
#            "analyze": "/api/v1/dreams/analyze",
#            "health": "/health"
#        }
#    }
#
#@app.get("/health")
#def health_check():
#    """Detailed health check"""
#    return {
#        "status": "healthy",
#        "ai_status": "loaded",
#        "vector_db": "ready",
#        "knowledge_chunks": len(knowledge_chunks),
#        "timestamp": datetime.now().isoformat()
#    }
#
#@app.post("/api/v1/dreams/analyze", response_model=DreamResponse)
#def analyze_dream(request: DreamRequest):
#    """
#    Analyze a dream and return psychological insights
#    """
#    try:
#        # Validate input
#        if not request.dreamText or len(request.dreamText) < 10:
#            raise HTTPException(status_code=400, detail="Dream text too short (minimum 10 characters)")
#        
#        # Analyze dream
#        result = analyze_dream_with_rag(request.dreamText, vectorstore, llm)
#        
#        # Return response
#        return DreamResponse(
#            emotions=result['nlp_features']['emotions'],
#            symbols=result['nlp_features']['symbols'],
#            entities=result['nlp_features']['entities'],
#            analysisShort=result['analysis_short'],
#            analysisFull=result['analysis_full'],
#            sourcesUsed=result['sources_used']
#        )
#        
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
#
## Run with: uvicorn main:app --host 0.0.0.0 --port 8000
