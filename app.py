import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI PDF Chatbot")

st.write("Upload a PDF and ask questions about its contents.")

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Read the PDF
    pdf_reader = PdfReader(uploaded_file)

    # Count pages
    number_of_pages = len(pdf_reader.pages)

    st.info(f"📄 Number of pages: {number_of_pages}")

    # Extract text
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Show extracted text
    st.subheader("📖 Extracted Text")

    st.text_area(
        "PDF Content",
        text,
        height=400
    )