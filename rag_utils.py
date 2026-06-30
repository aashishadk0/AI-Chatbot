import os
import nest_asyncio
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse

nest_asyncio.apply()
load_dotenv()

UPLOAD_DIR = Path("data/uploads")
PARSED_DIR = Path("data/parsed")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdfs(uploaded_files):
    saved_files = []

    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        saved_files.append(file_path)

    return saved_files


def parse_pdf_with_llama(file_path):
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True
    )

    documents = parser.load_data(str(file_path))

    parsed_text = ""

    for doc in documents:
        parsed_text += doc.text + "\n\n"

    parsed_path = PARSED_DIR / f"{file_path.stem}.md"

    with open(parsed_path, "w", encoding="utf-8") as f:
        f.write(parsed_text)

    return parsed_path


def get_parsed_pdfs():
    return [file.name for file in PARSED_DIR.glob("*.md")]


def read_parsed_pdf(filename):
    file_path = PARSED_DIR / filename

    if not file_path.exists():
        return ""

    return file_path.read_text(encoding="utf-8")


def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

CHROMA_DIR = Path("data/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection_name(pdf_name):
    clean_name = pdf_name.replace(".md", "").replace(" ", "_").replace("-", "_")
    return f"pdf_{clean_name}".lower()


def create_vector_store(pdf_name, chunks):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = get_collection_name(pdf_name)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return True


def retrieve_chunks(pdf_name, question, top_k=3):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = get_collection_name(pdf_name)

    collection = client.get_collection(name=collection_name)

    question_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    retrieved_chunks = results["documents"][0]

    return retrieved_chunks


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are a PDF question-answering assistant.

Answer the question using only the context below.
If the answer is not available in the context, say:
"The answer is not available in the selected PDF."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You answer questions only from the provided PDF context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content