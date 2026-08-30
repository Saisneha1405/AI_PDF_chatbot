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


st.title("AI PDF Chatbot")
st.write("Upload a PDF and process its content.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"{uploaded_file.name} uploaded successfully!")

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    # Extract text
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Check whether text was extracted
    if not text.strip():
        st.error("Could not extract text from this PDF.")
    else:
        st.success("PDF text extracted successfully!")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(text)

        embeddings = embedding_model.encode(chunks)
        # Convert embeddings to NumPy array
        embeddings = np.array(embeddings).astype("float32")


        # Create FAISS index
        embedding_dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(embedding_dimension)
        index.add(embeddings)

        st.success(f"FAISS index created with {index.ntotal} vectors!")
        st.subheader("Test PDF Search")

        query = st.text_input("Ask something about your PDF:")

        if query:
            # Convert question into an embedding
            query_embedding = embedding_model.encode([query])

            # Convert to FAISS-compatible format
            query_embedding = np.array(query_embedding).astype("float32")

            # Search FAISS for the most relevant chunks
            distances, indices = index.search(query_embedding, k=3)

            # Retrieve the actual text chunks
            retrieved_chunks = [chunks[idx] for idx in indices[0]]

            # Combine chunks into context for the AI model
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

        st.success(
            f"Generated embeddings for {len(embeddings)} chunks!"
        )

        st.write(
            f"Embedding dimensions: {embeddings.shape[1]}"
        )

        st.success(f"PDF divided into {len(chunks)} text chunks.")

        # Show basic information
        st.write(f"**Pages:** {len(pdf_reader.pages)}")
        st.write(f"**Characters extracted:** {len(text)}")
        st.write(f"**Text chunks created:** {len(chunks)}")

        # Preview
        with st.expander("Preview extracted text"):
            st.write(text[:3000])

        with st.expander("Preview first text chunk"):
            st.write(chunks[0])