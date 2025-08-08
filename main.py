from fastapi import FastAPI, Query
import httpx
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain_core.prompts import PromptTemplate

# ---------------- LOAD ENV ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

# ---------------- CONFIG ----------------
config = {
    "agent_name": "SunnyBot",
    "agent_persona": "a friendly customer support agent who loves helping people with warmth and emojis",
    "greeting_options": [
        "Hi there! 😊",
        "Hello! 👋",
        "Hey friend! 🌟"
    ],
    "unknown_answer": "I'm not sure about that 🤔, but I can connect you with a real representative for more help.",
    "chatgpt_model": "gpt-4o-mini",
    "retriever_k": 2
}

# ---------------- SAMPLE FAQ DATA ----------------
faq_data = [
    {"question": "What are your opening hours?", "answer": "We're open from 9 AM to 9 PM every day! 🕘"},
    {"question": "Where are you located?", "answer": "We’re at 123 Sunshine Street, Pleasantville! 📍"},
    {"question": "Do you offer home delivery?", "answer": "Yes! We deliver within a 10km radius 🚚"},
    {"question": "How can I contact support?", "answer": "You can email us at support@sunnybot.com or call 📞 +123456789."},
    {"question": "Do you accept credit cards?", "answer": "Yes, we accept all major credit and debit cards 💳"},
    {"question": "Do you have vegan options?", "answer": "Absolutely! We have a variety of vegan-friendly dishes 🌱"},
    {"question": "Where are your products manufactured", "answer": "They are made in the USA"}
]

# ---------------- VECTOR DB SETUP ----------------
documents = [Document(page_content=f"Q: {item['question']} A: {item['answer']}") for item in faq_data]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": config["retriever_k"]})

# ---------------- PROMPT TEMPLATE ----------------
prompt_template = """
You are {agent_name}, {agent_persona}

If the answer is in the context, respond accurately and warmly.
If not found, say: "{unknown_answer}"

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["agent_name", "agent_persona", "unknown_answer", "context", "question"]
)

# ---------------- CHATGPT CALL ----------------
async def call_chatgpt(prompt_text: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": config["chatgpt_model"],
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

# ---------------- RAG CHAIN ----------------
async def rag_answer(question: str) -> str:
    docs = await retriever.ainvoke(question)  # updated from aget_relevant_documents
    context = " ".join([doc.page_content for doc in docs])

    if not context.strip():
        return config["unknown_answer"]

    prompt_text = prompt.format(
        agent_name=config["agent_name"],
        agent_persona=config["agent_persona"],
        unknown_answer=config["unknown_answer"],
        context=context,
        question=question
    )
    return await call_chatgpt(prompt_text)

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="SunnyBot RAG Service")

@app.get("/")
async def root():
    return {"message": f"{config['agent_name']} is ready to assist you! 💬"}

@app.get("/ask")
async def ask_customer(question: str = Query(..., description="Customer's question")):
    answer = await rag_answer(question)
    return {"question": question, "answer": answer}
