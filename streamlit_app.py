import streamlit as st
import os
import time
from dotenv import load_dotenv
from groq import Groq
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup as Soup

# 1. Configuration Setup (Supports Local AND Cloud)
# Load dotenv locally
if os.path.exists(".env"):
    load_dotenv(".env")

def get_secret(key):
    # Try Streamlit Secrets (Cloud) first, then Environment Variables (Local)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.environ.get(key)

PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# 2. Embedding Engine Initialization (Cloud Pinecone Inference)
@st.cache_resource
def get_embeddings():
    if not PINECONE_API_KEY:
        st.error("❌ Missing PINECONE_API_KEY in Secrets/Environment!")
        return None
    # Multi-lingual cloud embeddings (No local download!)
    return PineconeEmbeddings(model="multilingual-e5-large", pinecone_api_key=PINECONE_API_KEY)

# 3. Sidebar & Header UI
st.set_page_config(page_title="Sukkur IBA Assistant", page_icon="🎓", layout="wide")
st.image("https://www.iba-suk.edu.pk/assets/images/iba_logo.png", width=120)
st.title("🎓 Sukkur IBA | Admission & Policy Assistant")

with st.sidebar:
    st.header("⚙️ Admin Dashboard")
    uni_url = st.text_input("University URL", value="https://www.iba-suk.edu.pk/")
    if st.button("🚀 Update Knowledge Base"):
        with st.status("Cleaning and Indexing in Cloud (Steadily)...", expanded=True) as s:
            try:
                if not PINECONE_API_KEY: raise ValueError("PINECONE_API_KEY is not set!")
                
                # Deep Search Depth 2
                loader = RecursiveUrlLoader(url=uni_url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").get_text())
                raw_docs = loader.load()
                # Use standard splitting
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(raw_docs)
                
                # Sanitize metadata to remove None/null values for Pinecone
                for doc in docs:
                    doc.metadata = {k: v for k, v in doc.metadata.items() if v is not None}
                
                # Vector Search Initialization
                embeddings = get_embeddings()
                if not embeddings: return
                
                vector_store = PineconeVectorStore(
                    index_name=PINECONE_INDEX_NAME, 
                    embedding=embeddings, 
                    pinecone_api_key=PINECONE_API_KEY
                )
                
                # IMPORTANT: Batch your indexing to avoid (429) Too Many Requests
                batch_size = 50
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    vector_store.add_documents(batch)
                    time.sleep(1) # Controlled pacing for the free tier
                
                s.update(label="✅ Deep Index Complete!", state="complete")
                st.balloons()
            except Exception as e:
                s.update(label=f"❌ Error: {e}", state="error")

# 4. Chat Interface
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages: 
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Cloud Context..."):
            try:
                if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY is not set!")
                
                # 1. Direct Search in Pinecone
                embeddings = get_embeddings()
                if not embeddings: raise ValueError("Embeddings failed to initialize.")
                
                vector_store = PineconeVectorStore(
                    index_name=PINECONE_INDEX_NAME, 
                    embedding=embeddings, 
                    pinecone_api_key=PINECONE_API_KEY
                )
                search_results = vector_store.similarity_search(prompt, k=5)
                context = "\n\n".join([doc.page_content for doc in search_results])
                
                # 2. Direct Call to Groq
                client = Groq(api_key=GROQ_API_KEY)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are an IBA Assistant. Use only this context: {context}"},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1
                )
                
                response = chat_completion.choices[0].message.content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
