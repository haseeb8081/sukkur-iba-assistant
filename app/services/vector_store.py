from langchain_community.vectorstores import SupabaseVectorStore
import os
from supabase.client import Client, create_client
from app.core.config import settings

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

# Initialize embeddings using your Local Llama 2 file
from langchain_community.embeddings import LlamaCppEmbeddings

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "llama-2-7b-chat.ggmlv3.q4_0.bin")

try:
    embeddings = LlamaCppEmbeddings(model_path=model_path, n_ctx=2048)
    print(f"✅ Using Local Llama 2 file: {model_path}")
except Exception as e:
    # Fallback to free HuggingFace model just in case llama-cpp fails
    print(f"⚠️ Local model load error: {e}. Falling back to HuggingFace...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    return SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
