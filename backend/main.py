from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sqlite3
import math
from collections import Counter
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
import simplemma

# Preparación de tokenización y stopwords
TOKEN_RE = re.compile(r"\b[a-záéíóúñ]{3,25}\b", re.IGNORECASE) # re.IGNORECASE para mayúsculas/minúsculas
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("spanish"))
try:
    LANG_DATA = simplemma.load_data("es")
except Exception:
    LANG_DATA = "es"

def norm_text(s):
    return unicodedata.normalize("NFKC", s).lower()

def tokenize(text):
    text = norm_text(text)
    tokens = TOKEN_RE.findall(text)
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        try:
            lemma = simplemma.lemmatize(t, LANG_DATA)
        except Exception:
            lemma = t
        out.append(lemma.lower())
    return out

# Configuración de la API y base de datos
BASE = Path(__file__).parent
DB_FILE = BASE / "data" / "processed" / "index.sqlite"
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

conn_sqlite = None
if DB_FILE.exists():
    conn_sqlite = sqlite3.connect(str(DB_FILE), check_same_thread=False, timeout=30)
    conn_sqlite.executescript("""
        PRAGMA cache_size = -20000;
        PRAGMA mmap_size = 3000000000;
        PRAGMA temp_store = MEMORY;
    """)

@app.get("/status")
def status():
    if not conn_sqlite:
        return {"backend": None}
    c = conn_sqlite.cursor()
    c.execute("SELECT COUNT(*), COUNT(DISTINCT file) FROM docs")
    docs, files = c.fetchone()
    c.execute("SELECT COUNT(*) FROM terms")
    terms = c.fetchone()[0] 
    return {
        "backend": "sqlite",
        "docs_loaded": docs,
        "files_processed": files,
        "terms_loaded": terms,
    }

# Endpoint de búsqueda
@app.get("/search")
def search(query: str, top_k: int = 10):
    if not conn_sqlite:
        return {"results": [], "error": "Índice no cargado"}
    
    # Procesar query
    terms = tokenize(query)
    if not terms:
        return {"results": []}
    c = conn_sqlite.cursor()

    # buscamos los terminos de la query en la tabla terms
    unique_terms = list(set(terms))
    placeholders = ",".join("?" * len(unique_terms))
    c.execute(
        f"SELECT term, id, idf FROM terms WHERE term IN ({placeholders})", # Obtener IDF de términos de la query
        unique_terms,
    )

    # Obtener información de términos (id, idf) para términos encontrados
    term_info = {row[0]: {"id": row[1], "idf": row[2] or 1.0} for row in c.fetchall()}
    if not term_info:
        return {"results": []}
    # Calcular frecuencia de términos en la query
    qtf = Counter(terms)

    # Calcular pesos del query y su norma
    wq_term = {}
    idf_by_tid = {}
    for t in qtf:
        info = term_info.get(t)
        if info is not None:
            idf_val = float(info["idf"])
            wq_term[t] = qtf[t] * idf_val
            idf_by_tid[int(info["id"])] = idf_val
    
    q_norm = math.sqrt(sum(w * w for w in wq_term.values())) or 1.0


    # Recuperar postings de los términos en la query 
    tids = []
    wq_by_tid = {}
    for term, info in term_info.items():
        tid = int(info["id"])
        tids.append(tid)
        wq_by_tid[tid] = wq_term.get(term, 0.0)
    
    placeholders = ",".join("?" * len(tids))
    
    # Usar tfidf precalculado y norma precalculada
    c.execute(
        f"""
        SELECT p.term_id, p.doc_id, p.tfidf, d.norm, d.file, d.snippet
        FROM postings p
        JOIN docs d ON p.doc_id = d.id
        WHERE p.term_id IN ({placeholders})
        """,
        tids,
    )
    rows = c.fetchall()
    if not rows:
        return {"results": []}
    
    docs_scores = {}
    docs_info = {}

    for tid, did, tfidf_val, norm_val, file_path, snippet in rows:
        # Usar tfidf precalculado 
        w_d = tfidf_val if tfidf_val else 0.0
        w_q = wq_by_tid.get(tid, 0.0)
        
        if w_q == 0.0 or w_d == 0.0:
            continue
        #para cada documento, acumular el score, numerador de la similitud del coseno y la info
        docs_scores[did] = docs_scores.get(did, 0.0) + (w_d * w_q)
        docs_info[did] = (file_path, snippet, norm_val)

    results = []

    # Calcular similitud del coseno
    for did, num in docs_scores.items():
        file_path, snippet, doc_norm = docs_info[did]
        doc_norm = doc_norm if doc_norm and doc_norm > 0 else 1.0 #asegurar no división por cero

        # Calcular denominador de la similitud del coseno
        denom = doc_norm * q_norm
        
        
        score = float(num) / denom if denom > 0 else 0.0
        
        
        results.append((score, did, file_path, snippet))
    
    results.sort(key=lambda x: (-x[0], x[1])) # ordenar por score descendente, luego por id ascendente
    top_results = results[:top_k]

    return {
        "query_terms": terms,
        "results": [
            {
                "doc_id": doc_id,
                "score": round(score, 6),
                "file": file_path,
                "text": snippet,
            }
            for score, doc_id, file_path, snippet in top_results
        ],
    }


