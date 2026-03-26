from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastembed import TextEmbedding
from supabase import create_client
from PyPDF2 import PdfReader
import anthropic
import io
import re
from typing import Optional

from config import get_settings
from models import SearchRequest, SearchResponse, ChunkResult

# --- Init ---

settings = get_settings()
app = FastAPI(title="RetailDocs RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(settings.supabase_url, settings.supabase_key)
claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)

# Lazy loading du modèle d'embeddings (évite timeout healthcheck)
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=settings.embedding_model)
    return _embedder


# --- Helpers ---


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = re.split(r"\s+", text.strip())
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# --- Routes ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    retailer: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    store: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    doc_type: Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")

    pdf_bytes = await file.read()
    text = extract_text_from_pdf(pdf_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Impossible d'extraire du texte de ce PDF")

    # 1. Créer le document
    doc_result = (
        supabase.table("documents")
        .insert(
            {
                "filename": file.filename,
                "retailer": retailer,
                "country": country,
                "store": store,
                "year": year,
                "doc_type": doc_type,
            }
        )
        .execute()
    )
    document_id = doc_result.data[0]["id"]

    # 2. Découper en chunks
    chunks = chunk_text(text, settings.chunk_size)

    # 3. Générer les embeddings
    embeddings = [e.tolist() for e in get_embedder().embed(chunks)]

    # 4. Insérer les chunks avec embeddings
    rows = [
        {
            "document_id": document_id,
            "content": chunk,
            "chunk_index": i,
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    supabase.table("chunks").insert(rows).execute()

    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunks_count": len(chunks),
    }


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    # 1. Embedding de la question
    query_embedding = list(get_embedder().embed([req.question]))[0].tolist()

    # 2. Recherche vectorielle via la fonction Supabase
    result = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": settings.top_k,
            "filter_retailer": req.retailer,
            "filter_country": req.country,
            "filter_store": req.store,
            "filter_year": req.year,
            "filter_doc_type": req.doc_type,
        },
    ).execute()

    if not result.data:
        return SearchResponse(answer="Aucun document pertinent trouvé.", sources=[])

    # 3. Construire le contexte pour Claude
    context_parts = []
    sources = []
    for match in result.data:
        context_parts.append(
            f"[Source: {match['filename']} | {match['retailer']} | {match['country']}]\n{match['content']}"
        )
        sources.append(
            ChunkResult(
                content=match["content"][:300],
                similarity=round(match["similarity"], 4),
                filename=match["filename"],
                retailer=match.get("retailer"),
                country=match.get("country"),
                store=match.get("store"),
                year=match.get("year"),
                doc_type=match.get("doc_type"),
            )
        )

    context = "\n\n---\n\n".join(context_parts)

    # 4. Appel Claude API
    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""Tu es un assistant spécialisé dans l'analyse de documents retail (merchandising, guidelines, rapports de visite).
Réponds à la question en te basant uniquement sur le contexte fourni. Si le contexte ne permet pas de répondre, dis-le clairement.
Réponds en français.

Contexte:
{context}

Question: {req.question}""",
                }
            ],
        )
        answer = message.content[0].text
    except Exception as e:
        answer = f"Erreur Claude API : {e}"

    return SearchResponse(answer=answer, sources=sources)


@app.get("/documents")
def list_documents(
    retailer: Optional[str] = None,
    country: Optional[str] = None,
    year: Optional[int] = None,
):
    query = supabase.table("documents").select("*")
    if retailer:
        query = query.eq("retailer", retailer)
    if country:
        query = query.eq("country", country)
    if year:
        query = query.eq("year", year)
    result = query.order("created_at", desc=True).execute()
    return result.data


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    supabase.table("documents").delete().eq("id", document_id).execute()
    return {"deleted": document_id}


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
