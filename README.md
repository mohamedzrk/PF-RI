# Motor de Búsqueda - Práctica Final RI

El objetivo principal de esta práctica ha sido construir un motor de búsqueda capaz de indexar y recuperar información relevante sobre una colección documental de gran volumen (superior a 10 GB).

La arquitectura se divide en:
- **Backend (FastAPI):** Encargado del procesamiento de lenguaje natural, la lógica de indexación y la gestión de consultas.
- **Frontend (ReactJS):** Una aplicación de una sola página que permite al usuario interactuar con el motor de búsqueda.
- **Almacenamiento (SQLite):** Una base de datos relacional optimizada para albergar un índice invertido escalable.

## Objetivos y limitaciones

Los objetivos a cumplir en esta práctica han sido:

- Procesar una colección documental superior a los **10 GB** de texto bruto teniendo en cuenta las limitaciones de mi máquina (procesador Intel Core i3 de 2 núcleos y 8GB de RAM).
- Implementar un flujo de trabajo que incluya análisis léxico, normalización, tokenización mediante expresiones regulares, filtrado de stopwords y lematización.
- Aplicar la ponderación **TF-IDF** y el cálculo de la **Similitud del Coseno** para garantizar que los resultados recuperados sean los más afines a la intención de búsqueda del usuario.

## Corpus

He usado el subconjunto monolingüe en español del corpus **CC-100** (Monolingual Datasets from Web Crawl Data). Este corpus fue creado con el propósito de recrear el conjunto de datos necesario para el entrenamiento del modelo de lenguaje a gran escala XLM-R.

*(Para más información sobre la implementación leer la memoria entregada junto a la practica)*


## Instrucciones de instalación

**Nota:** Debido al tamaño del corpus y del índice, los archivos de datos no están en el repositorio y deben añadirse manualmente (el link para la descarga de estos se encuentra en la memoria entregada en prado)

### 1. Configuración del Backend

Desde la terminal, entra en la carpeta `backend`:

```bash
cd backend
```

**Instalación de dependencias:**


#  Instalar librerías
```bash
pip install -r requirements.txt
```

**Preparación de los datos:**

Antes de arrancar, debes colocar los archivos en las carpetas correspondientes:

1.  Añade el corpus en formato CC-100 (`es.txt`) en la carpeta:
    `/backend/data/raw/docs/`
2.  Configura el índice:
    - **Si ya tienes el índice (`index.sqlite`):** Añádelo a la carpeta `/backend/data/processed/`.
    - **Si NO tienes el índice:** Ejecuta el script de construcción (requiere que el corpus esté en la carpeta raw y pulsando control + c al cargar algunos documentos, estos documentos se guardan y podemos hacer pruebas con indices mas pequeños):
      ```bash
      python build_index_sqlite.py
      ```

**Iniciar el servidor:**

```bash
uvicorn main:app --reload
```

### 2. Configuración del Frontend

Abre otra terminal y navega a la carpeta del buscador:

```bash
cd frontend/buscador-ri
```

Instala las dependencias e inicia la aplicación:

```bash
npm install
npm start
```

La aplicación se abrirá en `http://localhost:3000`.
