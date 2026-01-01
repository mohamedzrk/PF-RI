import sqlite3
import re
import math
import unicodedata
from pathlib import Path
from collections import Counter

import nltk
from nltk.corpus import stopwords
import simplemma

# Rutas y parámetros básicos
BASE = Path(__file__).parent
RAW_DIR = BASE / "data" / "raw" / "docs"
OUT_DIR = BASE / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = OUT_DIR / "index.sqlite"


LEER_BYTES = 16 * 1024 * 1024
CHUNK_TAM = 5000000 #tamaño máximo de documento
LOTE = 1000 #número de documentos por lote
TERM_CHUNK = 2000 # para inserciones en BD
POST_CHUNK = 20000 # para inserciones en BD

# Expresiones regulares para tokenizar y separar documentos
TOKEN_RE = re.compile(r"\b[a-záéíóúñ]{3,25}\b", re.IGNORECASE)
SEP_RE = re.compile(r"\r?\n\r?\n+")

# Carga de stopwords y lematizador
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

# Normalizacion, tokenización, eliminacion de 
# palabras vacías y lematización
def tokenize(text):
    text = norm_text(text)
    tokens = TOKEN_RE.findall(text)
    lemmas = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        try:
            lemma = simplemma.lemmatize(token, LANG_DATA)
        except Exception:
            lemma = token
        lemmas.append(lemma.lower())
    return lemmas


# Creamos las tablas necesarias en la base de datos
#con las configuraciones adecuadas para rendimiento
def preparar_bd(conn):
    c = conn.cursor()
    c.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;
        CREATE TABLE IF NOT EXISTS docs(
            id INTEGER PRIMARY KEY,
            file TEXT,
            snippet TEXT,
            length INTEGER,
            norm REAL DEFAULT 1.0
        );
        CREATE TABLE IF NOT EXISTS terms(
            id INTEGER PRIMARY KEY,
            term TEXT UNIQUE,
            df INTEGER DEFAULT 0,
            idf REAL DEFAULT 1.0
        );
        CREATE TABLE IF NOT EXISTS postings(
            term_id INTEGER,
            doc_id INTEGER,
            tf INTEGER,
            tfidf REAL DEFAULT 0,
            PRIMARY KEY(term_id, doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term_id);
        CREATE INDEX IF NOT EXISTS idx_postings_doc ON postings(doc_id);
    """)
    conn.commit()


# Procesa un lote de documentos, actualizando la base de datos
def procesar_lote(textos, refs, conn, term_cache):
    if not textos:
        return 0
    c = conn.cursor()
    datos = []
    vocab = set()

    # Tokenizar cada documento y calcular TF
    for texto, ref in zip(textos, refs): # zip para pares texto-ref
        lemmas = tokenize(texto)
        if not lemmas:
            continue
        tf = Counter(lemmas)
        datos.append((ref, " ".join(lemmas)[:400], sum(tf.values()), tf)) # aqui guardamos el archivo, snippet, la longitud y el tf
        vocab.update(tf.keys())# actualizamos el vocabulario con los términos del documento
    if not datos:
        return 0
    
    # Insertar términos nuevos en la tabla terms usando vocabulario
    nuevos = [t for t in vocab if t not in term_cache]
    if nuevos:
        c.executemany(
            "INSERT OR IGNORE INTO terms(term, df) VALUES(?, 0)",
            ((t,) for t in nuevos),
        )
        conn.commit()

        # Recuperar los IDs de los términos recién insertados y actualizar la caché
        for i in range(0, len(nuevos), TERM_CHUNK):
            chunk = nuevos[i:i+TERM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            c.execute(f"SELECT term, id FROM terms WHERE term IN ({placeholders})", chunk)
            for term, tid in c.fetchall():
                term_cache[term] = tid

    # Insertar documentos y postings
    postings = []
    df_delta = {} 
    for file, snippet, length, tf_dict in datos:
        # Insertar documento
        c.execute(
            "INSERT INTO docs(file, snippet, length) VALUES(?, ?, ?)",
            (file, snippet, length),
        )
        doc_id = c.lastrowid

        # creamos postings 
        for term, freq in tf_dict.items():
            tid = term_cache.get(term)
            if tid is None:
                c.execute("SELECT id FROM terms WHERE term = ?", (term,))
                row = c.fetchone()
                if row:
                    tid = row[0]
                    term_cache[term] = tid
                else:
                    continue
            postings.append((tid, doc_id, freq))
            df_delta[tid] = df_delta.get(tid, 0) + 1

# Insertar postings en lotes en la base de datos
    for i in range(0, len(postings), POST_CHUNK):
        chunk = postings[i:i+POST_CHUNK]
        c.executemany(
            "INSERT OR REPLACE INTO postings(term_id, doc_id, tf) VALUES(?, ?, ?)",
            chunk,
        )
# Actualizar df de los términos
    if df_delta:
        c.executemany(
            "UPDATE terms SET df = df + ? WHERE id = ?",
            ((v, k) for k, v in df_delta.items()),
        )
    return len(datos)

def procesar_fichero(path, conn):
    term_cache = {}
    processed = 0
    textos = []
    refs = []
    buf = ""

    print("Procesando:", path.name)

# Función para procesar y guardar el lote actual
    def commit_batch():
        nonlocal processed, textos, refs
        if textos:
            added = procesar_lote(textos, refs, conn, term_cache)
            conn.commit()
            processed += added
            print(f"Processed {processed} docs")
            textos, refs = [], []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(LEER_BYTES)

                # Si no hay más datos, procesar el resto del buffer
                if not chunk:
                    partes = [p for p in SEP_RE.split(buf) if p and p.strip()] # separamos docs finales 

                    # para cada doc 
                    for p in partes:
                        if len(textos) >= LOTE: # si llegamos al tamaño de lote, guardamos
                            commit_batch()
                        if len(p) <= CHUNK_TAM:
                            textos.append(p)
                            refs.append(str(path.relative_to(BASE)))
                        else:
                            # dividir en trozos si es demasiado grande
                            for i in range(0, len(p), CHUNK_TAM):
                                textos.append(p[i:i+CHUNK_TAM])
                                refs.append(str(path.relative_to(BASE)))
                    break


                # si no es el final, añadimos al buffer y separamos
                buf += chunk
                parts = SEP_RE.split(buf) 

                
                if len(parts) > 1:
                    completos = parts[:-1] # documentos completos
                    buf = parts[-1]  # ultimo fragmento incompleto 

                    # Procesar documentos completos
                    for doc in completos:
                        if not doc.strip():
                            continue
                        if len(textos) >= LOTE:
                            commit_batch()
                        if len(doc) <= CHUNK_TAM:
                            textos.append(doc)
                            refs.append(str(path.relative_to(BASE)))
                        else:
                            for i in range(0, len(doc), CHUNK_TAM):
                                textos.append(doc[i:i+CHUNK_TAM])
                                refs.append(str(path.relative_to(BASE)))

    # Manejo de interrupción para guardar el lote actual
    except KeyboardInterrupt:
        print("\nInterrumpido: guardando lote actual…")
        commit_batch()
        print("Índice parcial guardado. Saliendo.")
        return processed
    if textos:
        commit_batch()

    print("Terminado", path.name, "Total procesados:", processed)
    return processed


# Finaliza el índice calculando IDF, TF-IDF y normas de documentos
def finalizar_idf(conn):
    c = conn.cursor()
    print("Finalizando: calculando IDF…")

    # Contar número total de documentos
    c.execute("SELECT COUNT(*) FROM docs")
    N = c.fetchone()[0] or 1
    print("Docs totales:", N)

    # Calcular IDF: 1 + log(N / (df + 1))
    c.execute("SELECT id, df FROM terms")
    rows = c.fetchall()
    updates = []
    for tid, df in rows:
        if df > 0:
            idf = 1.0 + math.log(N / (df + 1))
        else:
            idf = 1.0
        updates.append((idf, tid))

# Actualizar IDF en la tabla terms
    if updates:
        c.executemany("UPDATE terms SET idf = ? WHERE id = ?", updates)
        conn.commit()


    # Calcular TF-IDF y norma para cada documento
    c.execute("SELECT id, idf FROM terms")
    idf_map = {tid: idf for tid, idf in c.fetchall()}

    print("Calculando TF-IDF y norma de documentos…")
    
    # Obtener todos los doc_ids
    c.execute("SELECT DISTINCT doc_id FROM postings ORDER BY doc_id")
    all_docs = [row[0] for row in c.fetchall()]
    
    # para cada documento, calcular tfidf y norma
    for doc_id in all_docs:
        c.execute("SELECT term_id, tf FROM postings WHERE doc_id = ?", (doc_id,))
        postings = c.fetchall()
        
        norm_squared = 0.0
        tfidf_updates = []
        
        for tid, tf in postings:
            idf = idf_map.get(tid, 1.0) # obtener idf
            tfidf = tf * idf
            tfidf_updates.append((tfidf, tid, doc_id))
            norm_squared += tfidf * tfidf
        
        # Actualizar tfidf de los postings
        c.executemany(
            "UPDATE postings SET tfidf = ? WHERE term_id = ? AND doc_id = ?",
            tfidf_updates
        )
        
        # Actualizar norma del documento
        doc_norm = math.sqrt(norm_squared) if norm_squared > 0 else 1.0
        c.execute("UPDATE docs SET norm = ? WHERE id = ?", (doc_norm, doc_id))
        
        if doc_id % 1000 == 0: # cada 1000 documentos, hacer commit
            conn.commit()
            print(f"Procesados {doc_id} documentos...")
    
    conn.commit()
    print("Índice finalizado.")



def main():
    
    archivos = sorted([p for p in RAW_DIR.glob("*") if p.is_file()])
    if not archivos:
        print("No hay archivos en", RAW_DIR)
        return
    if DB_FILE.exists(): 
        DB_FILE.unlink()
    conn = sqlite3.connect(str(DB_FILE))
    try:
        preparar_bd(conn)
        total = 0
        for f in archivos:
            total += procesar_fichero(f, conn)
        print("Docs procesados (totales):", total)
        finalizar_idf(conn)
    finally:
        conn.close()
        print("Índice creado en:", DB_FILE)

if __name__ == "__main__":
    main()
