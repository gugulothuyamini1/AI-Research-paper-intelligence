import re
import requests
import fitz
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Paper Intelligence",
    page_icon="📚",
    layout="wide"
)


OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:latest"


# ============================================================
# TITLE
# ============================================================

st.title("📚 AI Research Paper Intelligence")

st.caption(
    "Chat with your research papers using RAG + Ollama"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

ollama_url = st.sidebar.text_input(
    "Ollama Server URL",
    value=OLLAMA_URL
)

model = st.sidebar.text_input(
    "Ollama Model",
    value=MODEL
)

top_k = st.sidebar.slider(
    "Number of relevant pages",
    1,
    10,
    5
)


# ============================================================
# TEST OLLAMA
# ============================================================

if st.sidebar.button("🔌 Test Ollama"):

    try:

        response = requests.get(
            f"{ollama_url.rstrip('/')}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        installed_models = [
            item["name"]
            for item in data.get("models", [])
        ]

        st.sidebar.success(
            "✅ Ollama is running"
        )

        if installed_models:

            st.sidebar.write(
                "Installed models:"
            )

            for installed_model in installed_models:

                st.sidebar.write(
                    f"• {installed_model}"
                )

        else:

            st.sidebar.warning(
                "No models installed."
            )

    except Exception as error:

        st.sidebar.error(
            f"Ollama error: {error}"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "documents" not in st.session_state:

    st.session_state.documents = []


if "file_names" not in st.session_state:

    st.session_state.file_names = []


# ============================================================
# CLEAR CHAT
# ============================================================

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "📂 Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]{3,}\b",
            text.lower()
        )
    )


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    pdf = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text(
            "text"
        ).strip()

        pages.append(
            {
                "page_number": page_number,
                "text": text
            }
        )

    pdf.close()

    return {
        "filename": uploaded_file.name,
        "pages": pages
    }


# ============================================================
# PROCESS UPLOADS
# ============================================================

if uploaded_files:

    current_file_names = [
        file.name
        for file in uploaded_files
    ]

    if current_file_names != st.session_state.file_names:

        documents = []

        for uploaded_file in uploaded_files:

            try:

                document = process_pdf(
                    uploaded_file
                )

                documents.append(
                    document
                )

            except Exception as error:

                st.error(
                    f"Could not process "
                    f"{uploaded_file.name}: "
                    f"{error}"
                )

        st.session_state.documents = documents

        st.session_state.file_names = (
            current_file_names
        )


# ============================================================
# DOCUMENTS
# ============================================================

if st.session_state.documents:

    st.success(
        f"✅ "
        f"{len(st.session_state.documents)} "
        f"research paper(s) loaded."
    )

    with st.expander(
        "📄 Uploaded Research Papers"
    ):

        for document in (
            st.session_state.documents
        ):

            st.write(
                f"**{document['filename']}** "
                f"— "
                f"{len(document['pages'])} pages"
            )


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_relevant_pages(
    documents,
    question,
    k
):

    question_words = tokenize(
        question
    )

    results = []

    for document in documents:

        for page in document["pages"]:

            page_text = page["text"]

            page_words = tokenize(
                page_text
            )

            common_words = (
                question_words
                & page_words
            )

            score = len(
                common_words
            )

            results.append(
                {
                    "score": score,
                    "filename": document[
                        "filename"
                    ],
                    "page_number": page[
                        "page_number"
                    ],
                    "text": page_text
                }
            )

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:k]


# ============================================================
# CREATE CONTEXT
# ============================================================

def create_context(
    retrieved_pages
):

    context = []

    for page in retrieved_pages:

        text = page["text"]

        if not text:

            text = (
                "No selectable text "
                "was found on this page."
            )

        context.append(
            f"""
SOURCE: {page['filename']}
PAGE: {page['page_number']}

{text[:7000]}
"""
        )

    return "\n\n".join(
        context
    )


# ============================================================
# ASK OLLAMA
# ============================================================

def ask_ollama(
    question,
    documents,
    history
):

    # --------------------------------------------------------
    # RETRIEVE RELEVANT PAGES
    # --------------------------------------------------------

    retrieved_pages = (
        retrieve_relevant_pages(
            documents,
            question,
            top_k
        )
    )

    # --------------------------------------------------------
    # CREATE RAG CONTEXT
    # --------------------------------------------------------

    context = create_context(
        retrieved_pages
    )

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are an AI research paper assistant.

Answer questions using only the supplied
research paper content.

Rules:

1. Do not invent information.

2. If the information is not present,
   say that the uploaded papers do not
   contain enough information.

3. Cite information using:

   [filename, p. N]

4. Give clear and accurate answers.

5. Explain technical concepts simply
   when appropriate.

6. When summarizing a paper, include:
   - Objective
   - Methodology
   - Results
   - Conclusion
   - Limitations
   - Future work
"""

    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
RESEARCH PAPER CONTENT:

{context}

QUESTION:

{question}

Answer using the research paper content.
Include page citations.
"""

    # --------------------------------------------------------
    # MESSAGES
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add recent conversation
    messages.extend(
        history[-6:]
    )

    messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    # --------------------------------------------------------
    # OLLAMA API
    # --------------------------------------------------------

    api_url = (
        f"{ollama_url.rstrip('/')}"
        f"/api/chat"
    )

    response = requests.post(
        api_url,
        json={
            "model": model,
            "messages": messages,
            "stream": False
        },
        timeout=600
    )

    # --------------------------------------------------------
    # ERROR CHECK
    # --------------------------------------------------------

    if response.status_code != 200:

        raise Exception(
            f"Ollama returned "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    if "message" not in data:

        raise Exception(
            f"Unexpected Ollama response: "
            f"{data}"
        )

    return data["message"]["content"]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT
# ============================================================

if st.session_state.documents:

    question = st.chat_input(
        "Ask a question about your research paper..."
    )

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                question
            )

        # ----------------------------------------------------
        # AI MESSAGE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🔎 Searching research papers..."
            ):

                try:

                    answer = ask_ollama(
                        question,
                        st.session_state.documents,
                        st.session_state.messages[:-1]
                    )

                    st.markdown(
                        answer
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        """
                        ❌ Cannot connect to Ollama.

                        Make sure Ollama is running:

                        `ollama serve`
                        """
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "⏳ Ollama took too long to respond."
                    )

                except Exception as error:

                    st.error(
                        f"❌ {error}"
                    )

else:

    st.info(
        "📂 Upload one or more PDF research papers "
        "to start chatting."
    )