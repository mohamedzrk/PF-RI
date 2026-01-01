import { useState } from "react";
import "./App.css";


export default function App() {
  const [query, setQuery] = useState(""); 
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

  // Función para realizar la búsqueda llamando al backend
  const buscar = async () => {
    setError("");
    setResults([]);
    if (!query.trim()) {
      setError("Escribe una consulta.");
      return;
    }
    try {
      const res = await fetch("http://127.0.0.1:8000/search?query=" + encodeURIComponent(query));
      if (!res.ok) throw new Error("error servidor");
      const data = await res.json();
      setResults(data.results || []);
    } catch {
      setError("No se pudo conectar con el backend.");
    }
  };

  return (
    <div className="app">
      <h1>Buscador RI</h1>

      <div className="search">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Escribe una consulta..."
          onKeyDown={(e) => e.key === "Enter" && buscar()}
        />
        <button onClick={buscar}>Buscar</button>
      </div>

      {error && <p>{error}</p>}

      <h3>Resultados</h3>
      <ul>
        {results.map((r, i) => (
          <li key={i}>
            <div>doc_id: {r.doc_id} | score: {r.score}</div>
            <div>file: {r.file}</div>
            <div>{r.text}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
