import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS # FAISS
from langchain.text_splitter import CharacterTextSplitter # Chunking
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

genai.configure(api_key=os.getenv("GOOGLE-API-KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Configure Embedding Model sentence-transformers/all-MiniLM-L6-v2
@st.cache_resource(show_spinner = "Loading the Model.....")
def myembedding_model():
    return (HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"))

embedding_model = myembedding_model()

# Reading the PDF after the Front End
st.header("RAG using :blue[HF Embeddings + FAISS db]")
uploaded_file = st.file_uploader("Upload the Document",type=["pdf"])

if uploaded_file:
    raw_text = ""
    pdf =PdfReader(uploaded_file)
    for index, page in enumerate(pdf.pages):
        context = page.extract_text()
        if context:
            raw_text += context
    
    # Chunking using schema
    if raw_text.strip():
        document = Document(page_content = raw_text)
        splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        chunks = splitter.split_documents([document])
        
        # HF Embedding
        texts = [chunk.page_content for chunk in chunks]
        vector_db = FAISS.from_texts(texts,embedding_model)
        retriever = vector_db.as_retriever()
        st.markdown("Document Processed Successfully✅. Ask Questions Below")
        user_input = st.text_input("Enter your query....")
        
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Analysing the document...."):
                retrieved_doc = retriever.get_relevant_documents(user_input)
                context = "\n\n".join(doc.page_content for doc in retrieved_doc)

                prompt = f""" You are an expert assistant. Use the context below to answer the query. If unsure, just say - I don't know.
                Context: {context},
                User Query: {user_input}
                Answer: """
                response = model.generate_content(prompt)
                st.markdown("Answer: ")
                st.write(response.text)
else:
    st.warning("Please Upload the PDF for review and analysis")




