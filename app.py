import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.embeddings import EmbeddingManager
from src.ingestion import process_single_pdf, split_documents
from src.rag_chain import get_rag_stream
from src.registry import (
    file_hash, load_registry, save_registry,
    is_processed, register_file, remove_file,
)
from src.retriever import RAGRetriever
from src.vectorstore import VectorStore
from quiz import generate_quiz

load_dotenv()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")


# ---------- cached resources (only load once per session, not on every rerun) ----------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_manager():
    return EmbeddingManager()


@st.cache_resource(show_spinner="Connecting to vector store...")
def get_vectorstore():
    return VectorStore()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_llm(temperature: float):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found. Create a `.env` file (see `.env.example`) "
            "with your key, then restart the app."
        )
        st.stop()
    return ChatGroq(api_key=api_key, model_name="openai/gpt-oss-20b", temperature=temperature, max_tokens=5000)


embedding_manager = get_embedding_manager()
vectorstore = get_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "registry" not in st.session_state:
    st.session_state.registry = load_registry()


# ---------------------------- sidebar ----------------------------

with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
        help="Already-processed files are detected by content and skipped automatically.",
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            file_bytes = uploaded.getvalue()
            h = file_hash(file_bytes)

            if is_processed(h, st.session_state.registry):
                st.info(f"'{uploaded.name}' already processed — skipping.")
                continue

            save_path = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(file_bytes)

            with st.status(f"Processing {uploaded.name}...", expanded=True) as status:
                st.write("Reading PDF...")
                documents = process_single_pdf(save_path)

                st.write("Splitting into chunks...")
                chunks = split_documents(documents)

                if not chunks:
                    status.update(label=f"No usable text found in {uploaded.name}", state="error")
                    continue

                st.write(f"Embedding {len(chunks)} chunks...")
                texts = [c.page_content for c in chunks]
                embeddings = embedding_manager.generate_embeddings(texts)

                st.write("Storing in vector database...")
                vectorstore.add_documents(chunks, embeddings)

                st.session_state.registry = register_file(uploaded.name, h, len(chunks), st.session_state.registry)
                save_registry(st.session_state.registry)

                status.update(label=f"'{uploaded.name}' ready ({len(chunks)} chunks)", state="complete")

    st.divider()
    st.subheader("Processed files")
    if not st.session_state.registry:
        st.caption("No files processed yet.")
    else:
        for fname, info in list(st.session_state.registry.items()):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{fname}**  \n:gray[{info['chunk_count']} chunks · {info['processed_at']}]")
            if col2.button("🗑️", key=f"del_{fname}", help=f"Remove {fname}"):
                vectorstore.delete_by_source(fname)
                st.session_state.registry = remove_file(fname, st.session_state.registry)
                save_registry(st.session_state.registry)
                uploaded_path = Path(UPLOAD_DIR) / fname
                if uploaded_path.exists():
                    uploaded_path.unlink()
                st.rerun()

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 5)
    min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.2, 0.05)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    if st.button("🧹 Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------- main chat area ----------------------------

st.title("📚 RAG Chat")
st.caption("Ask questions about the documents you've uploaded.")

if not st.session_state.registry:
    st.warning("Upload at least one PDF from the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Sources (confidence: {msg.get('confidence', 0):.0%})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — page {s['page']} — score `{s['score']:.2f}`")
                    st.caption(s["preview"])

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    llm = get_llm(temperature)

   # ---------- QUIZ ----------
# Make sure query is a string
query = query or ""

is_quiz = any(
    word in query.lower()
    for word in ["quiz", "mcq", "multiple-choice", "multiple choice"]
)

if is_quiz:

    # Read the number of questions requested by the user
    match = re.search(
        r"(\d+)\s*(?:questions?|mcqs?|multiple[- ]choice questions?)",
        query.lower()
    )

    if match:
        num_questions = int(match.group(1))
    else:
        num_questions = 10

    # Keep the requested number within a safe range
    num_questions = max(1, min(num_questions, 50))

    with st.chat_message("assistant"):
        try:
            data = vectorstore.collection.get(
                include=["documents"]
            )

            documents = data.get("documents", [])

            if not documents:
                st.error("No document content found. Please upload a PDF first.")

            else:
                context = "\n\n".join(documents)

                quiz = generate_quiz(
                    context,
                    llm,
                    num_questions=num_questions
                )

                st.session_state.quiz = quiz
                st.session_state.quiz_submitted = False

                st.success(
                    f"{len(quiz)}-question quiz generated!"
                )

        except Exception as e:
            st.error(f"Could not generate quiz: {e}")

    # ---------- NORMAL RAG CHAT ----------
        else:   
         retriever = RAGRetriever(vectorstore, embedding_manager)

        with st.chat_message("assistant"):
            token_stream, sources, confidence = get_rag_stream(
                query,
                retriever,
                llm,
                top_k=top_k,
                min_score=min_score
            )

            answer = st.write_stream(token_stream)

            if sources:
                with st.expander(
                    f"📄 Sources (confidence: {confidence:.0%})"
                ):
                    for s in sources:
                        st.markdown(
                            f"**{s['source']}** — page {s['page']} "
                            f"— score `{s['score']:.2f}`"
                        )
                        st.caption(s["preview"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence": confidence,
        })
        # ---------- DISPLAY QUIZ ----------

if "quiz" in st.session_state and st.session_state.quiz:

    quiz = st.session_state.quiz

    st.header(f"📝 {len(quiz)}-Question Quiz")

    with st.form("quiz_form"):

        for i, question in enumerate(quiz, 1):

            st.markdown(
                f"### {i}. {question['question']}"
            )

            options = question["options"]

            st.radio(
                "Choose your answer:",
                ["A", "B", "C", "D"],
                format_func=lambda x: f"{x}) {options[x]}",
                key=f"quiz_answer_{i}"
            )

        submitted = st.form_submit_button("✅ Submit Quiz")

    if submitted:    
        score = 0

        if "topic_stats" not in st.session_state:
            st.session_state.topic_stats = {}

        for i, question in enumerate(quiz, 1):
            selected = st.session_state[f"quiz_answer_{i}"]
            correct = question["answer"]
            topic = question.get("topic", "General")

            if topic not in st.session_state.topic_stats:
                st.session_state.topic_stats[topic] = {
                    "correct": 0,
                    "wrong": 0
                }

            if selected == correct:
                score += 1
                st.session_state.topic_stats[topic]["correct"] += 1
            else:
                st.session_state.topic_stats[topic]["wrong"] += 1

        st.session_state.quiz_submitted = True
        st.session_state.quiz_score = score   
    # ---------- RESULTS ----------

    if st.session_state.get("quiz_submitted", False):

        score = st.session_state.quiz_score
        total = len(quiz)
        percentage = score / total * 100

        st.success(
            f"🎉 Your Score: {score}/{total} ({percentage:.0f}%)"
        )
        # ---------- PERSONALIZED RECOMMENDATIONS ----------

if "topic_stats" in st.session_state:

    weak_topics = []

    for topic, stats in st.session_state.topic_stats.items():

        if stats["wrong"] >= 2:
            weak_topics.append(topic)

    st.subheader("🎯 Personalized Recommendations")

    if weak_topics:

        st.warning(
            "You should practice these weak topics:"
        )

        # Create retriever for topic-based recommendations
        retriever = RAGRetriever(
            vectorstore,
            embedding_manager
        )

        for topic in weak_topics:

            stats = st.session_state.topic_stats[topic]

            st.markdown(
                f"### 📌 {topic}"
            )

            st.markdown(
                f"You got **{stats['wrong']} incorrect answer(s)** "
                f"from this topic."
            )

            # Retrieve study material related to the weak topic
            topic_results = retriever.retrieve(
                topic,
                top_k=3,
                score_threshold=0.0
            )

            if topic_results:

                st.success(
                    f"📚 Study material found for {topic}"
                )

                for result in topic_results:

                    st.markdown(
                        f"**Source:** "
                        f"{result['metadata'].get('source_file', 'Uploaded document')}"
                    )

                    st.caption(
                        result["content"][:500] + "..."
                    )

                st.info(
                    f"💡 Recommendation: Review the above material "
                    f"and practice {topic} again."
                )

            else:

                st.info(
                    f"No specific study material was found for {topic}."
                )

        st.divider()

        st.info(
            "🎯 Personalized learning path: "
            "Focus on your weak topics first, review the relevant "
            "study material, and then take another practice quiz."
        )

    else:

        st.success(
            "🎉 No major weak topics detected. "
            "Keep practicing to maintain your performance!"
        )

        st.subheader("Answer Review")

        for i, question in enumerate(quiz, 1):

            selected = st.session_state[f"quiz_answer_{i}"]
            correct = question["answer"]

            if selected == correct:
                st.success(
                    f"**{i}. Correct!** "
                    f"You selected {selected}) {question['options'][selected]}"
                )
            else:
                st.error(
                    f"**{i}. Incorrect.** "
                    f"You selected {selected}) {question['options'][selected]}  \n"
                    f"Correct answer: {correct}) {question['options'][correct]}"
                )