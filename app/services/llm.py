from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.services.vector_store import get_vector_store

def get_llm():
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL_NAME,
        temperature=0.1
    )

def get_rag_chain():
    llm = get_llm()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    template = """
    You are an AI assistant for a university department. Your goal is to answer student questions accurately based on the provided context.
    
    Context: {context}
    Question: {question}
    
    Rules for response:
    1. If the answer is not in the context, say "I'm sorry, I don't have that information. Please contact the department via email."
    2. Give helpful, professional, and friendly answers.
    3. If there is a URL in the context related to the answer, include it.
    
    Answer:"""
    
    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
