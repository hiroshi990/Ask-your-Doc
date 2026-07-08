import time
from narwhals.typing import Backend
import streamlit as st
from sympy.polys.polyconfig import query
from client import BackendAPIError, ask_question,upload_file,paste_text,delete

st.set_page_config(
    page_title="Ask Your Doc",
    layout="wide"
)

import streamlit as st

st.markdown(
    """
<style>
/* ---------- Global ---------- */
.block-container {
    padding-top: 1.2rem;
    max-width: 1050px;
}

#MainMenu, footer {
    visibility: hidden;
}

header {
    background: transparent;
}

body {
    color: #2F4A28;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFF1D2 0%, #FFF7E8 100%);
    border-right: 1px solid #E3D3AE;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #3D5A32;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    background: #FFF8E8;
    border: 1.5px dashed #D9C59A;
    border-radius: 22px;
    padding: 1rem;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #8FA65F, #6F8A43);
    color: white;
    border: none;
    border-radius: 999px;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(111, 138, 67, 0.22);
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #7F9852, #627D3A);
    color: white;
    transform: translateY(-1px);
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    background: #FFF0C9;
    border: 1px solid #E5C98F;
    border-radius: 999px;
    box-shadow: 0 8px 25px rgba(140, 110, 50, 0.13);
}

[data-testid="stChatInput"] textarea {
    color: #2F4A28;
}

/* ---------- Question box ---------- */
.question-box {
    background: linear-gradient(135deg, #FFF5D9, #F7F1D9);
    border: 1px solid #E3D3AE;
    border-radius: 20px;
    padding: 18px 22px;
    margin: 22px 0 12px 0;
    color: #2F4A28;
    font-size: 16px;
    box-shadow: 0 4px 14px rgba(100, 85, 40, 0.06);
}

/* ---------- Answer box ---------- */
.answer-box {
    background: #FFFFFF;
    border: 1px solid #E3D3AE;
    border-left: 7px solid #7A944F;
    border-radius: 22px;
    padding: 24px;
    margin: 12px 0 8px 0;
    color: #263A22;
    font-size: 16px;
    line-height: 1.65;
    box-shadow: 0 10px 30px rgba(80, 100, 60, 0.08);
}

/* ---------- Expander / citations ---------- */
[data-testid="stExpander"] {
    background: #FFF8EA;
    border: 1px solid #E3D3AE;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 6px 18px rgba(100, 85, 40, 0.06);
}

.streamlit-expanderHeader {
    color: #3D5A32;
    font-weight: 700;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: 16px;
}

/* ---------- Uploaded document cards ---------- */
.doc-card {
    background: #FFF8E8;
    border: 1px solid #E3D3AE;
    border-radius: 16px;
    padding: 10px 12px;
    margin-bottom: 8px;
    color: #2F4A28;
    box-shadow: 0 4px 12px rgba(100, 85, 40, 0.05);
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #FFF8E8;
}

::-webkit-scrollbar-thumb {
    background: #C8B783;
    border-radius: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-title-row">
            <span class="hero-leaf">🍃</span>
            <span class="hero-title">Ask Your Doc!</span>
        </div>
        <div class="hero-subtitle">
            Chat with your documents and get cited answers.
        </div>
    </div>

    <style>
    .hero-wrapper {
        text-align: center;
        margin-top: 0.7rem;
        margin-bottom: 2.5rem;
    }

    .hero-title-row {
        display: flex;
        align-items: center;
        margin-left:-25px;
        justify-content: center;
        gap: 12px;
    }

    .hero-leaf {
        font-size: 44px;
        line-height: 1;
        transform: translateY(3px) rotate(-8deg);
    }

    .hero-title {
        color: #33542a;
        font-size: 58px;
        font-weight: 700;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #5E6B55;
        font-size: 21px;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type = [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
        ".html", ".htm", ".md", ".txt", ".csv", ".json", ".xml",
        ".tiff", ".tif", ".webp"],
        accept_multiple_files = True
    )

    if st.button(
        "Process documents",
        type = "primary",
        disabled = not uploaded_files,
        use_container_width=False
    ):
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = upload_file(uploaded_file)
                info = {
                    "document_id":result.get("document",{}).get("document_id"),
                    "document_name":result.get("document",{}).get("document_name"),
                }
                
                if info not in st.session_state.uploaded_documents:
                    st.session_state.uploaded_documents.append(
                        info
                    )
                st.success(f"Uploaded {info.get('document_name')}")
            
            except BackendAPIError as e :
                st.error(str(e))
    st.divider()
    text_input = st.text_area(
        "Text Input",
        max_chars = 2000,
        placeholder = "Feed custom text as context...",

    )

    if st.button(
        "Process Text",
        type="primary",
        disabled= not text_input):

        try:
            with st.spinner("Processing text..."):
                text_result = paste_text(text_input)
            info = {
                "document_id":text_result.get("document",{}).get("document_id"),
                "document_name":text_result.get("document",{}).get("document_name"),
            }
            if info not in st.session_state.uploaded_documents:
                st.session_state.uploaded_documents.append(info)
            st.success(f"Uploaded text!")
        except BackendAPIError as e:
            st.error(str(e))

    if st.session_state.uploaded_documents:
        st.subheader("Uploaded Documents",text_alignment="center")
        for doc in st.session_state.uploaded_documents:
            col1 , col2 = st.columns([0.78,0.22])
            with col1:
                st.write(f"📄 {doc['document_name']}")
            with col2:
                if st.button(
                    "🗑️",
                    help=f"Delete {doc['document_name']}",):

                    try:
                        delete(doc)
                        st.session_state.uploaded_documents = [
                            d for d in st.session_state.uploaded_documents
                            if d["document_id"] != doc["document_id"]
                        ]

                        st.success("Deleted")
                    except BackendAPIError as e:
                        st.error(str(e))
    if st.session_state.messages:    
        if st.button("Clear conversation",use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# Existing Messages

for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(
            f"""
            <div class="question-box">
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="answer-box">
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        metadata = message.get('metadata')  
        citations = metadata.get("citations",[])

        if citations:
            with st.expander("View Citations"):
                for index, citation in enumerate(citations, start =1):
                    st.markdown(
                        f"""
                        **[{index}] {citation.get("document_name","Unknown document")}**
                        - {citation.get("section_title","")}
                        - Excerpt : {citation.get("excerpt","")}
                        """
                    )
            


# new question

question = st.chat_input("Chat with your documents")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content":question
        }
    )
    st.markdown(
        f"""
        <div class="question-box">
            {question}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    try:
        with st.spinner("Finding best answer..."):
            start = time.perf_counter()
            response = ask_question(query=question)
            client_latency_ms = (time.perf_counter() - start) * 1000
        answer = response.get("answer")

        metadata = {
            "latency_ms": client_latency_ms,
            "citations": response.get("citations",[]),
            "faithfulness" : response.get("faithfulness_score",None),
            "relevancy": response.get("answer_relevancy",None)
        }

        st.markdown(
            f"""
        <div class="answer-box">
            {answer}
        </div>
        """,
        unsafe_allow_html=True,
        )

        citations = metadata.get("citations",[])

        if citations:
            with st.expander("View Citations"):
                for index, citation in enumerate(citations, start =1):
                    st.markdown(
                        f"""
                        **[{index}] {citation.get("document_name","Unknown document")}**\n
                        - **{citation.get("section_title","")}**\n
                        - Excerpt : {citation.get("excerpt","")}
                        """
                    )

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content": answer,
                "metadata":metadata
            }
        )
    except BackendAPIError as exc:
        st.error(str(exc))
    
                