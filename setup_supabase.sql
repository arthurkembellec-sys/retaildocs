-- Activer l'extension pgvector
create extension if not exists vector;

-- Table des documents uploadés
create table if not exists documents (
    id uuid default gen_random_uuid() primary key,
    filename text not null,
    retailer text,
    country text,
    store text,
    year integer,
    doc_type text,
    created_at timestamptz default now()
);

-- Table des chunks avec embeddings
create table if not exists chunks (
    id uuid default gen_random_uuid() primary key,
    document_id uuid references documents(id) on delete cascade,
    content text not null,
    chunk_index integer not null,
    embedding vector(384),
    created_at timestamptz default now()
);

-- Index pour la recherche vectorielle
create index if not exists chunks_embedding_idx
    on chunks using hnsw (embedding vector_cosine_ops);

-- Fonction de recherche par similarité avec filtres
create or replace function match_chunks(
    query_embedding vector(384),
    match_count int default 5,
    filter_retailer text default null,
    filter_country text default null,
    filter_store text default null,
    filter_year int default null,
    filter_doc_type text default null
)
returns table (
    chunk_id uuid,
    document_id uuid,
    content text,
    chunk_index int,
    similarity float,
    filename text,
    retailer text,
    country text,
    store text,
    year int,
    doc_type text
)
language plpgsql
as $$
begin
    return query
    select
        c.id as chunk_id,
        c.document_id,
        c.content,
        c.chunk_index,
        1 - (c.embedding <=> query_embedding) as similarity,
        d.filename,
        d.retailer,
        d.country,
        d.store,
        d.year,
        d.doc_type
    from chunks c
    join documents d on d.id = c.document_id
    where
        (filter_retailer is null or d.retailer = filter_retailer)
        and (filter_country is null or d.country = filter_country)
        and (filter_store is null or d.store = filter_store)
        and (filter_year is null or d.year = filter_year)
        and (filter_doc_type is null or d.doc_type = filter_doc_type)
    order by c.embedding <=> query_embedding
    limit match_count;
end;
$$;
