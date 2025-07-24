# RAG-Based
Deploying RAG

🧠 RAG-Powered PDF Q&A App

-This Streamlit app implements a Retrieval-Augmented Generation (RAG) pipeline using:

-HuggingFace Embeddings (all-MiniLM-L6-v2)

-FAISS Vector Store

-Google Gemini 2.0 Flash Model

-LangChain for text processing

-Streamlit for the front end

-It allows users to upload a PDF document, processes the text, and then enables contextual Q&A over that document.

🚀 Features

-🔍 Extracts text from uploaded PDFs

-✂️ Splits large documents into chunks

-🧠 Embeds text with sentence-transformers/all-MiniLM-L6-v2

-📦 Stores embeddings in a FAISS vector database

-🤖 Uses Google Gemini (2.0 Flash) for answering user queries

-💬 Simple Streamlit interface with interactive Q&A


📷 App Interface

-Upload a PDF document.

-The app extracts and chunks the content.

-Ask questions related to the PDF content.
-Get answers powered by retrieved context + Gemini.
