-- 1. Enable the pgvector extension to work with embeddings
create extension if not exists vector;

-- 2. Create a table to store your documents and their embeddings
create table if not exists documents (
  id bigserial primary key,
  content text, -- corresponds to Document.page_content
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(384) -- 384 for all-mini-LM-L6-v2; use 1536 for OpenAI
);

-- 3. Create a function to search for documents based on embedding similarity
create or replace function match_documents (
  query_embedding vector(384), -- Must match the embedding dimension above
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
