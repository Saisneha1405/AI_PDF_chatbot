# AI PDF Chatbot

A Streamlit web application that lets you upload a PDF and ask natural-language questions about its content. The app uses **Retrieval-Augmented Generation (RAG)** to find the most relevant sections of a document before generating an answer with Google Gemini.

---

## Project Overview

Instead of sending an entire PDF to a language model on every question, this chatbot:

1. Extracts and indexes the document locally
2. Retrieves the most relevant text chunks for each question
3. Sends only that context to Gemini for a grounded answer

Answers are intended to come from the uploaded PDF, not from the model's general knowledge.

---

## Features

- **PDF upload** — Upload text-based PDF files through a Streamlit file uploader
- **Text extraction** — Extract selectable text from PDF pages using PyPDF
- **Semantic chunking** — Split documents into overlapping chunks for better retrieval
- **Vector search** — Build a FAISS index for fast similarity search over document chunks
- **RAG-powered Q&A** — Retrieve top relevant chunks and generate answers with Gemini
- **Session caching** — Process each PDF once; reuse chunks and FAISS index across reruns
- **Chat history** — View previous questions and answers in the current session
- **Clear Chat** — Reset conversation history without re-processing the uploaded PDF
- **Document previews** — Inspect extracted text, chunk count, and page metadata
- **Error handling** — Graceful handling of Gemini API failures without crashing the app

---

## How the RAG Pipeline Works

```
PDF Upload
    ↓
Text Extraction (PyPDF)
    ↓
Text Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding Generation (SentenceTransformer: all-MiniLM-L6-v2)
    ↓
FAISS Vector Index
    ↓
User Question → Query Embedding → Top-3 Chunk Retrieval
    ↓
Prompt (Context + Question + Rules)
    ↓
Google Gemini (gemini-3.6-flash)
    ↓
Answer displayed in chat UI
```

### Pipeline details

| Step | Description |
|------|-------------|
| **Upload** | User selects a PDF via Streamlit |
| **Extract** | Text is read page by page from the PDF |
| **Chunk** | Text is split into 1000-character chunks with 200-character overlap |
| **Embed** | Each chunk is converted into a 384-dimensional vector |
| **Index** | Embeddings are stored in a FAISS `IndexFlatL2` index |
| **Retrieve** | The user's question is embedded and matched against the top 3 chunks |
| **Generate** | Retrieved context and the question are sent to Gemini with strict grounding rules |
| **Respond** | The answer is shown in the chat interface |

The model is instructed to answer only from the retrieved PDF context and to say *"I could not find the answer in the uploaded PDF."* when the information is not present.

---

## Technologies Used

| Category | Technology |
|----------|------------|
| **UI** | Streamlit |
| **PDF parsing** | PyPDF |
| **Text splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector search** | FAISS (CPU) |
| **Numerical computing** | NumPy |
| **LLM** | Google Gemini (`gemini-3.6-flash`) |
| **API client** | `google-genai` |
| **Configuration** | `python-dotenv` |

---

## Project Structure

```
AI_PDF_chatbot/
├── app.py                  # Main Streamlit application (RAG pipeline + chat UI)
├── requirements.txt        # Python dependencies
├── project_description.txt # Project documentation notes
├── .gitignore              # Ignored files (.env, .venv, PDFs, etc.)
├── .env                    # Local environment variables (create this yourself)
└── README.md               # Project documentation
```

> **Note:** The `.env` file is not committed to version control. Create it locally before running the app.

---

## Installation

### Prerequisites

- Python 3.10+ recommended
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd AI_PDF_chatbot
   ```

2. **Create and activate a virtual environment**

   **Windows (PowerShell):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install sentence-transformers
   ```

   > `sentence-transformers` is required by `app.py` for embedding generation.

---

## Configure `GEMINI_API_KEY`

1. Create a file named `.env` in the project root (same folder as `app.py`).

2. Add your API key:

   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. Replace `your_api_key_here` with your actual Gemini API key from Google AI Studio.

4. Do **not** commit the `.env` file to Git. It is already listed in `.gitignore`.

The application loads this variable at startup using `python-dotenv`:

```python
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
```

---

## Run the Application

With your virtual environment activated, run:

```bash
streamlit run app.py
```

**Windows alternative** (if `streamlit` is not recognized):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will open the app in your browser (typically at `http://localhost:8501`).

---

## How to Use the Chatbot

1. **Start the app** using the command above.
2. **Upload a PDF** using the file uploader.
3. **Wait for processing** — the app extracts text, creates chunks, builds embeddings, and indexes them in FAISS. This runs once per uploaded PDF.
4. **Ask a question** in the chat input about the document.
5. **Read the answer** in the chat interface. Previous questions and answers remain visible for the session.
6. **Ask follow-up questions** — the PDF is not re-processed; the cached FAISS index is reused.
7. **Clear Chat** — click the button to reset conversation history while keeping the uploaded PDF indexed.
8. **Upload a new PDF** — processing runs again and chat history resets for the new document.

### Tips

- Works best with **text-based PDFs** that contain selectable text.
- Scanned or image-only PDFs may fail text extraction.
- Use the expanders at the bottom to preview extracted text and the first chunk.

---

## Error Handling

The app includes error handling around the Gemini API call to prevent crashes and invalid chat entries.

| Scenario | Behavior |
|----------|----------|
| **High demand / service unavailable** (HTTP 429, 503) | Shows a friendly Streamlit warning; user can retry |
| **Other Gemini API errors** | Shows an error message with the API status code |
| **Unexpected errors** | Shows a generic error message; app continues running |
| **Empty model response** | Shows a warning; nothing is added to chat history |
| **Failed PDF text extraction** | Shows an error; session state is cleared |
| **Successful response** | User and assistant messages are saved to chat history |

Failed API calls do **not** add invalid assistant responses to the chat history.

---

## Future Improvements

- **OCR support** — Handle scanned and image-based PDFs
- **Page-level citations** — Show which page each answer came from
- **Persistent vector storage** — Save FAISS index to disk to avoid re-processing
- **Streaming responses** — Stream Gemini output for a better chat experience
- **Multi-PDF support** — Search across multiple uploaded documents
- **Sidebar settings** — Configurable chunk size, top-k retrieval, and model options
- **Deployment** — Publish on Streamlit Cloud or similar platform
- **Dependency cleanup** — Align `requirements.txt` with packages actually used in the app

---

## License

Add a license here if you plan to open-source or share this project.
