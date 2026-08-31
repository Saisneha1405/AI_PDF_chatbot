import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if "pdf_file_id" not in st.session_state:
    st.session_state.pdf_file_id = None
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.text = None
    st.session_state.page_count = None
    st.session_state.embedding_dim = None

st.title("AI PDF Chatbot")
st.write("Upload a PDF and process its content.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"{uploaded_file.name} uploaded successfully!")

    if uploaded_file.file_id != st.session_state.pdf_file_id:
        pdf_reader = PdfReader(uploaded_file)

        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error("Could not extract text from this PDF.")
            st.session_state.pdf_file_id = None
            st.session_state.chunks = None
            st.session_state.index = None
            st.session_state.text = None
            st.session_state.page_count = None
            st.session_state.embedding_dim = None
        else:
            st.success("PDF text extracted successfully!")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)

            embeddings = embedding_model.encode(chunks)
            embeddings = np.array(embeddings).astype("float32")

            embedding_dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dimension)
            index.add(embeddings)

            st.session_state.pdf_file_id = uploaded_file.file_id
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.text = text
            st.session_state.page_count = len(pdf_reader.pages)
            st.session_state.embedding_dim = embedding_dimension

            st.success(f"FAISS index created with {index.ntotal} vectors!")
            st.success(f"Generated embeddings for {len(chunks)} chunks!")
            st.success(f"PDF divided into {len(chunks)} text chunks.")

    if st.session_state.chunks is not None and st.session_state.index is not None:
        chunks = st.session_state.chunks
        index = st.session_state.index
        text = st.session_state.text

        st.subheader("Test PDF Search")

        query = st.text_input("Ask something about your PDF:")

        if query:
            query_embedding = embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")

            distances, indices = index.search(query_embedding, k=3)
            retrieved_chunks = [chunks[idx] for idx in indices[0]]
            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
            You are an AI assistant that answers questions based only on the uploaded PDF.

            Use the following PDF context to answer the question.

            PDF CONTEXT:
            {context}

            USER QUESTION:
            {query}

            Rules:
            - Answer only from the PDF context.
            - Do not make up information.
            - If the answer is not present in the context, say:
              "I could not find the answer in the uploaded PDF."
            - Give a clear and simple answer.
            """

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            st.subheader("Answer")
            st.write(response.text)

            st.subheader("Retrieved PDF Context")

            for i, chunk in enumerate(retrieved_chunks):
                st.write(f"### Result {i + 1}")
                st.write(chunk)
                st.divider()

        st.write(
            f"Embedding dimensions: {st.session_state.embedding_dim}"
        )

        st.write(f"**Pages:** {st.session_state.page_count}")
        st.write(f"**Characters extracted:** {len(text)}")
        st.write(f"**Text chunks created:** {len(chunks)}")

        with st.expander("Preview extracted text"):
            st.write(text[:3000])

        with st.expander("Preview first text chunk"):
            st.write(chunks[0])