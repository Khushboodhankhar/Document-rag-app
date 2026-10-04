"""Polished Streamlit workspace for a document-grounded RAG demonstration."""
import html
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.document_loader import DocumentLoader
from src.embeddings import OpenAIConfigurationError, get_api_key
from src.rag_chain import RAGChain, RAGResult
from src.vector_store import FAISSVectorStore

load_dotenv()
st.set_page_config(page_title="NIYANTRA | Document Intelligence", page_icon="◈", layout="wide")


def inject_styles() -> None:
    """Apply a bright, polished NIYANTRA workspace theme."""
    st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {background: radial-gradient(circle at 81% -12%, #123358 0, #080d18 38%, #070b14 100%); color: #e8eef8;}
    .block-container {max-width: 1380px; padding: 2rem 3rem 3.5rem;}
    h1, h2, h3 {color: #f6fbff; letter-spacing: -0.02em;}
    .brand {display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:0 0 1.5rem; border-bottom:1px solid rgba(148,180,211,.16); margin-bottom:2rem;}
    .brand-name {font-size:1.42rem; font-weight:800; letter-spacing:.16em; color:#66e3e5;}
    .brand-subtitle {font-size:.85rem; color:#91a3b9; margin-top:.22rem; letter-spacing:.02em;}
    .brand-tagline {font-size:.8rem; color:#b5c3d3; margin-top:.48rem;}
    .online {color:#a9f4d1; font-size:.82rem; white-space:nowrap; background:rgba(80,210,164,.09); border:1px solid rgba(80,210,164,.25); border-radius:999px; padding:.42rem .75rem;}
    .hero {position:relative; overflow:hidden; background:linear-gradient(135deg, rgba(17,43,71,.88), rgba(12,24,42,.76)); border:1px solid rgba(102,227,229,.22); border-radius:18px; padding:1.65rem 1.7rem; margin:1rem 0 1.15rem;}
    .hero:after {content:''; position:absolute; width:270px; height:270px; border-radius:999px; border:1px solid rgba(102,227,229,.13); right:-105px; top:-175px; box-shadow:0 0 0 28px rgba(102,227,229,.025), 0 0 0 56px rgba(102,227,229,.015);}
    .hero-title {font-size:1.55rem; font-weight:800; color:#f7fbff; letter-spacing:-.035em; margin:.3rem 0 .38rem;}
    .hero-copy {font-size:.91rem; line-height:1.55; color:#aebed0; max-width:600px;}
    .trust-row {display:flex; flex-wrap:wrap; gap:.48rem; margin-top:1rem;}
    .trust-chip {font-size:.69rem; font-weight:700; letter-spacing:.045em; color:#bceff0; padding:.34rem .52rem; background:rgba(82,222,218,.07); border:1px solid rgba(82,222,218,.19); border-radius:999px;}
    .eyebrow {font-size:.72rem; font-weight:700; letter-spacing:.11em; color:#66e3e5; text-transform:uppercase;}
    .lede {color:#9cacc1; font-size:1rem; margin-top:-.35rem;}
    .panel, .answer-card, .metric-card, .empty-state {background:rgba(18,29,47,.72); border:1px solid rgba(148,180,211,.16); border-radius:16px; box-shadow:0 16px 40px rgba(0,0,0,.18);}
    .panel {padding:1.25rem; margin-bottom:1rem;}
    .sidebar-shell {padding:.15rem .1rem;}
    .panel-title {font-size:1rem; font-weight:700; color:#f2f8ff; margin-bottom:.25rem;}
    .muted {color:#8fa1b7; font-size:.83rem;}
    .doc-row {display:flex; align-items:center; gap:.6rem; padding:.62rem 0; border-bottom:1px solid rgba(148,180,211,.10);}
    .doc-row:last-child {border-bottom:0; padding-bottom:0;}
    .doc-icon {width:26px; height:26px; display:flex; align-items:center; justify-content:center; background:rgba(73,219,222,.10); border-radius:7px; font-size:.63rem; color:#7ce7e8;}
    .doc-name {font-size:.84rem; color:#dce7f5; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
    .metric-card {padding:.85rem 1rem; min-height:87px;}
    .metric-label {font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#8fa1b7;}
    .metric-value {font-size:1.35rem; font-weight:750; color:#f4fbff; padding-top:.22rem;}
    .metric-ready {color:#84efc2;}
    .answer-card {padding:1.4rem 1.5rem; margin-top:1rem;}
    .answer-title {font-size:1.05rem; font-weight:750; color:#f7fbff;}
    .grounded {font-size:.68rem; color:#73ece4; border:1px solid rgba(75,225,217,.32); border-radius:999px; padding:.26rem .5rem; letter-spacing:.08em; font-weight:700;}
    .answer-copy {font-size:1rem; line-height:1.7; color:#dce7f5; margin-top:1rem; white-space:pre-wrap;}
    .source-caption {font-size:.71rem; color:#77e8e4; letter-spacing:.09em; font-weight:750;}
    .pipeline {display:flex; flex-wrap:wrap; align-items:center; gap:.35rem; padding:.85rem 0 1.15rem;}
    .pipeline-stage {display:flex; align-items:center; gap:.32rem; font-size:.68rem; font-weight:700; letter-spacing:.055em; color:#bceff0; padding:.42rem .55rem; border-radius:8px; border:1px solid rgba(75,225,217,.20); background:rgba(75,225,217,.06);}
    .pipeline-icon {font-size:.8rem;}
    .pipeline-line {width:18px; height:1px; background:rgba(105,150,180,.5);}
    .status-card {background:rgba(13,26,43,.78); border:1px solid rgba(115,236,228,.18); border-radius:13px; padding:1rem; margin-top:1rem;}
    .status-label {font-size:.69rem; color:#8fa1b7; font-weight:700; letter-spacing:.1em;}
    .status-ready {color:#8aefc2; font-weight:700; margin:.32rem 0 .7rem;}
    .workflow-card {background:rgba(18,29,47,.62); border:1px solid rgba(148,180,211,.14); border-radius:13px; padding:1rem; min-height:130px;}
    .workflow-number {color:#64e3e2; font-size:.72rem; font-weight:800; letter-spacing:.08em;}
    .workflow-title {color:#eef8ff; font-size:.92rem; font-weight:700; margin:.38rem 0;}
    .workflow-copy {color:#92a5bc; font-size:.8rem; line-height:1.45;}
    .empty-state {text-align:center; padding:2.2rem 1.5rem; margin:1.3rem 0;}
    .empty-icon {font-size:2rem; margin-bottom:.45rem;}
    .empty-title {font-size:1.16rem; font-weight:700; color:#f4f9ff; text-transform:uppercase; letter-spacing:.06em;}
    .empty-copy {color:#97a9be; margin:.4rem auto 0; max-width:380px;}
    .error-card {border-left:3px solid #ffbd67; background:rgba(106,70,21,.19); border-radius:9px; padding:.85rem 1rem; color:#f8e5c2;}
    .demo-label {font-size:.72rem; font-weight:750; letter-spacing:.10em; color:#91a3b9; margin:1.25rem 0 .55rem;}
    .demo-note {font-size:.78rem; color:#8799b1; margin:-.15rem 0 .7rem;}
    .stButton > button {border-radius:9px; font-weight:650; border:1px solid rgba(84,231,226,.35); background:#0c777d; color:#f5ffff; transition:all .15s ease;}
    .stButton > button:hover {background:#10929a; border-color:#7cf5ef; color:#fff;}
    .stTextInput input {background:#101b2d !important; color:#edf8ff !important; border:1px solid rgba(138,178,210,.24) !important; border-radius:10px !important; padding:.86rem .9rem !important; font-size:1rem !important;}
    .stTextInput input:focus {border-color:#4fdddc !important; box-shadow:0 0 0 1px #4fdddc !important;}
    [data-testid="stFileUploader"] {background:rgba(12,25,42,.62); border:1px dashed rgba(91,225,222,.36); border-radius:12px; padding:.45rem;}
    [data-testid="stExpander"] {background:rgba(13,23,39,.58); border:1px solid rgba(148,180,211,.13); border-radius:10px;}
    /* Bright NIYANTRA palette: blush pink, soft white, and mint green. */
    .stApp {background:radial-gradient(circle at 88% -9%,#d9fff0 0,#fff8fb 39%,#fff2f7 100%);color:#293640;}
    h1, h2, h3 {color:#26333c;}
    .brand {border-bottom-color:#f0dce6;}
    .brand-name {color:#ec4f90; font-weight:850;}
    .brand-subtitle {color:#65717c;}
    .brand-tagline {color:#86909b;}
    .online {color:#218967 !important;background:#e5fff3;border-color:#9ce8c5;}
    .hero {background:linear-gradient(130deg,#ffe2ef,#fff4f9 54%,#e8fff4);border-color:#f5b6cf;box-shadow:0 14px 35px rgba(233,100,153,.10);}
    .hero:after {border-color:rgba(66,192,143,.25);box-shadow:0 0 0 28px rgba(66,192,143,.07),0 0 0 56px rgba(245,128,174,.06);}
    .hero-title {color:#25343b;}
    .hero-copy {color:#596670;}
    .trust-chip {color:#27745b;background:#effff7;border-color:#a8eacb;}
    .eyebrow {color:#e64b8a;}
    .lede {color:#65717c;}
    .panel,.answer-card,.metric-card,.empty-state {background:rgba(255,255,255,.88);border-color:#f0dce6;box-shadow:0 13px 30px rgba(171,86,126,.10);}
    .panel-title,.answer-title {color:#2d3941;}
    .muted {color:#76818a;}
    .doc-row {border-bottom-color:#f4e4eb;}
    .doc-icon {background:#ffe2ee;color:#df4b88;}
    .doc-name,.answer-copy {color:#42505a;}
    .metric-label,.demo-label {color:#7a8790;}
    .metric-value {color:#2f3c45;}
    .metric-ready,.status-ready {color:#25966d;}
    .grounded {color:#278467;border-color:#9ce8c5;background:#ecfff5;}
    .source-caption {color:#278467;}
    .pipeline-stage {color:#b83f75;border-color:#f3b8d0;background:#fff2f7;}
    .pipeline-line {background:#e2bdcb;}
    .status-card {background:linear-gradient(145deg,#fff,#f4fff9);border-color:#b8edcf;}
    .status-label,.workflow-copy,.empty-copy,.demo-note {color:#69757f;}
    .workflow-card {background:rgba(255,255,255,.8);border-color:#f0dce6;box-shadow:0 8px 20px rgba(171,86,126,.06);}
    .workflow-number {color:#e64b8a;}
    .workflow-title,.empty-title {color:#34414a;}
    .error-card {border-left-color:#ef8e5b;background:#fff3e9;color:#9a542f;}
    .stButton > button {border-color:#f19ac0;background:#ec5b98;color:#fff;box-shadow:0 6px 12px rgba(236,91,152,.16);}
    .stButton > button:hover {background:#d94682;border-color:#d94682;color:#fff;transform:translateY(-1px);}
    .stTextInput input {background:#fff !important;color:#38454e !important;border-color:#edbdd1 !important;}
    .stTextInput input:focus {border-color:#eb609a !important;box-shadow:0 0 0 2px rgba(235,96,154,.16) !important;}
    [data-testid="stFileUploader"] {background:#fffafd;border-color:#e78eb8;}
    [data-testid="stExpander"] {background:#fff;border-color:#f0dce6;}
    /* Streamlit's nested uploader uses separate default colors, so style it explicitly. */
    [data-testid="stFileUploaderDropzone"] {background:linear-gradient(135deg,#e4fff5,#eefaff) !important;border:1.5px dashed #78cbb8 !important;}
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div {color:#4d6870 !important;}
    [data-testid="stFileUploaderDropzone"] button {background:#bceede !important;color:#236c5d !important;border:1px solid #73cdb5 !important;box-shadow:none !important;}
    [data-testid="stFileUploaderDropzone"] button:hover {background:#a7e5d2 !important;transform:none !important;}
    .paper-mark {position:relative;width:126px;height:92px;flex:0 0 126px;margin-right:1.1rem;}
    .paper {position:absolute;width:55px;height:70px;border-radius:8px;border:1px solid;box-shadow:0 9px 18px rgba(82,141,128,.12);}
    .paper:before {content:'';position:absolute;left:11px;right:11px;top:21px;height:2px;background:currentColor;box-shadow:0 9px 0 currentColor,0 18px 0 currentColor;opacity:.48;}
    .paper-pink {background:#ffe4ef;color:#e871a3;border-color:#f2a9c9;transform:rotate(-11deg);left:10px;top:12px;}
    .paper-blue {background:#e2f4ff;color:#5eadd2;border-color:#9ed8f0;transform:rotate(8deg);left:52px;top:4px;}
    .paper-mint {background:#e0faef;color:#57ad8b;border-color:#9cdfc4;left:34px;top:18px;}
    @media (max-width: 800px) {.paper-mark {display:none;}}
    @media (max-width: 800px) {.block-container {padding:1.2rem;} .brand {align-items:flex-start; flex-direction:column;}}
    </style>""", unsafe_allow_html=True)


def initialise_state() -> None:
    for key, value in {
        "rag_chain": None, "documents": [], "uploaded_document_names": [],
        "last_result": None, "last_question": "", "query_input": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def document_icon(filename: str) -> str:
    return {".pdf": "PDF", ".docx": "DOC", ".txt": "TXT"}.get(Path(filename).suffix.lower(), "FILE")


def render_error(title: str, message: str, details: Exception | None = None) -> None:
    st.markdown(f'<div class="error-card"><strong>⚠ {html.escape(title)}</strong><br>{html.escape(message)}</div>', unsafe_allow_html=True)
    if details:
        with st.expander("Technical details"):
            st.code(str(details))


def ingest(uploaded_files) -> bool:
    """Run real extraction, chunking, embedding, and FAISS indexing with visible progress."""
    try:
        get_api_key()
    except OpenAIConfigurationError as exc:
        render_error("OpenAI API key required", str(exc))
        return False
    upload_dir = Path(".uploads")
    upload_dir.mkdir(exist_ok=True)
    try:
        with st.status("Processing knowledge base...", expanded=True) as status:
            status.write("✓ Preparing uploaded files")
            for file in uploaded_files:
                (upload_dir / file.name).write_bytes(file.getbuffer())
            status.write("✓ Extracting text")
            documents = DocumentLoader().load_files([upload_dir / file.name for file in uploaded_files])
            if not documents:
                raise ValueError("No readable text was found in the selected files.")
            status.write("✓ Splitting into chunks")
            status.write("✓ Generating embeddings")
            store_path = Path("vectorstore")
            if store_path.exists():
                shutil.rmtree(store_path)
            store = FAISSVectorStore.create_from_documents(documents, store_path)
            status.write("✓ Building FAISS index")
            st.session_state.rag_chain = RAGChain(store)
            st.session_state.documents = documents
            st.session_state.uploaded_document_names = [file.name for file in uploaded_files]
            st.session_state.last_result = None
            status.update(label="Knowledge base ready", state="complete", expanded=False)
        return True
    except Exception as exc:
        st.session_state.rag_chain = None
        render_error("Document processing failed", "We could not prepare this knowledge base. Check the file format and configuration, then try again.", exc)
        return False


def render_sidebar() -> None:
    st.markdown('<div class="eyebrow">Knowledge Base</div><div class="panel-title">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    st.caption("Supported: PDF • DOCX • TXT")
    names = [file.name for file in uploaded_files] if uploaded_files else st.session_state.uploaded_document_names
    st.markdown('<div class="panel-title" style="margin-top:1rem">Uploaded Files</div>', unsafe_allow_html=True)
    if names:
        for name in names:
            st.markdown(f"<div class='doc-row'><div class='doc-icon'>{document_icon(name)}</div><div class='doc-name'>{html.escape(name)}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No documents selected yet.</div>', unsafe_allow_html=True)
    if st.button("Process Documents", type="primary", use_container_width=True, disabled=not uploaded_files):
        if ingest(uploaded_files):
            st.success("Documents indexed and ready for questions.")
    indexed = st.session_state.rag_chain is not None
    count, chunks = len(st.session_state.uploaded_document_names), len(st.session_state.documents)
    status_heading = "Knowledge Base Ready" if indexed else "INDEX STATUS"
    state_text = "● Indexed" if indexed else "● Waiting for documents"
    state_colour = "#25966d" if indexed else "#bc7d26"
    st.markdown(f"""<div class="status-card"><div class="status-label">{status_heading}</div><div class="status-ready" style="color:{state_colour}">{state_text}</div><div class="muted">Documents <span style="float:right;color:#34414a">{count}</span></div><div class="muted" style="margin-top:.38rem">Chunks <span style="float:right;color:#34414a">{chunks}</span></div><div class="muted" style="margin-top:.38rem">Vector Store <span style="float:right;color:#e64b8a">FAISS</span></div></div>""", unsafe_allow_html=True)


def render_metrics() -> None:
    ready = st.session_state.rag_chain is not None
    values = [("Documents", len(st.session_state.uploaded_document_names)), ("Chunks", len(st.session_state.documents)), ("Vector Store", "FAISS"), ("Status", "Ready" if ready else "Waiting")]
    for column, (label, value) in zip(st.columns(4), values):
        colour = " metric-ready" if label == "Status" and ready else ""
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value{colour}">{value}</div></div>', unsafe_allow_html=True)


def render_result(result: RAGResult, question: str) -> None:
    st.markdown(f'<div class="answer-card"><div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;"><div class="answer-title">ANSWER</div><span class="grounded">DOCUMENT-GROUNDED</span></div><div class="answer-copy">{html.escape(result.answer)}</div></div>', unsafe_allow_html=True)
    st.markdown("### Evidence used")
    st.caption(f"Semantic retrieval found {len(result.sources)} relevant source chunk(s) for: “{question}”")
    for index, source in enumerate(result.sources, start=1):
        location = f"Page {source['page']}" if source.get("page") else "No page metadata"
        with st.expander(f"{index}. {source['filename']} · {location} · Chunk {source['chunk_index']} · Relevance {source['score']:.1%}"):
            st.markdown(f"**Document:** {source['filename']}  \n**Page:** {source.get('page') or 'N/A'}  \n**Chunk:** {source['chunk_index']}  \n**Similarity:** {source['score']:.3f}")
            st.write(source["text"])
    st.markdown('<div class="trust-row"><span class="trust-chip">ANSWER GROUNDED IN RETRIEVED CONTEXT</span><span class="trust-chip">FAISS SEMANTIC SEARCH</span><span class="trust-chip">SOURCE-TRACEABLE</span></div>', unsafe_allow_html=True)


def render_demo_questions() -> None:
    questions = [
        "How many paid annual leave days do full-time employees receive?",
        "How far ahead should a leave request be submitted?",
        "How many days each week may eligible employees work remotely?",
        "What is the meal reimbursement limit during approved travel?",
        "What security measure is required on company accounts?",
        "What is the spacecraft maintenance policy?",
    ]
    st.markdown('<div class="demo-label">DEMO QUESTIONS</div>', unsafe_allow_html=True)
    st.markdown('<div class="demo-note">The last question is intentionally absent and should return the required fallback.</div>', unsafe_allow_html=True)
    for left, right in zip(questions[::2], questions[1::2]):
        first, second = st.columns(2)
        if first.button(left, key=f"demo_{left}", use_container_width=True):
            st.session_state.query_input = left
        if second.button(right, key=f"demo_{right}", use_container_width=True):
            st.session_state.query_input = right


inject_styles()
initialise_state()
try:
    get_api_key()
    service_ready, service_message = True, "● System Ready"
except OpenAIConfigurationError:
    service_ready, service_message = False, "● API Key Required"
service_colour = "#218967" if service_ready else "#bc7d26"
st.markdown('<div class="brand"><div><div class="brand-name">NIYANTRA</div><div class="brand-subtitle">Your document intelligence workspace</div><div class="brand-tagline">Upload. Retrieve. Verify.</div></div><div class="paper-mark" aria-label="Stack of documents"><span class="paper paper-pink"></span><span class="paper paper-blue"></span><span class="paper paper-mint"></span></div></div>', unsafe_allow_html=True)
left_column, main_column = st.columns([0.30, 0.70], gap="large")
with left_column:
    st.markdown('<div class="sidebar-shell">', unsafe_allow_html=True)
    render_sidebar()
    st.markdown("</div>", unsafe_allow_html=True)
with main_column:
    st.markdown('<div class="eyebrow">RAG Challenge Demonstration</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="eyebrow">Reliable answers, traceable evidence</div><div class="hero-title">Ask your documents with confidence.</div><div class="hero-copy">NIYANTRA retrieves the most relevant document chunks, then prepares a concise answer constrained to that evidence. Every response exposes its source file and location.</div><div class="trust-row"><span class="trust-chip">PDF · DOCX · TXT</span><span class="trust-chip">LOCAL FAISS INDEX</span><span class="trust-chip">STRICT RAG PROMPT</span></div></div>', unsafe_allow_html=True)
    render_metrics()
    st.markdown('<div class="pipeline"><span class="pipeline-stage"><span class="pipeline-icon">⇧</span>UPLOAD</span><span class="pipeline-line"></span><span class="pipeline-stage"><span class="pipeline-icon">✦</span>CHUNK</span><span class="pipeline-line"></span><span class="pipeline-stage"><span class="pipeline-icon">◌</span>EMBED</span><span class="pipeline-line"></span><span class="pipeline-stage"><span class="pipeline-icon">⌕</span>RETRIEVE</span><span class="pipeline-line"></span><span class="pipeline-stage"><span class="pipeline-icon">✧</span>GENERATE</span><span class="pipeline-line"></span><span class="pipeline-stage"><span class="pipeline-icon">✓</span>CITE</span></div>', unsafe_allow_html=True)
    if st.session_state.rag_chain is None:
        st.markdown('<div class="empty-state"><div class="empty-icon">⌁</div><div class="empty-title">Build your knowledge base</div><div class="empty-copy">Upload one or more PDF, DOCX, or TXT files in the left panel, then select <strong>Process Documents</strong> to create the local FAISS index.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="eyebrow" style="margin-top:1rem">Ask Your Documents</div>', unsafe_allow_html=True)
        render_demo_questions()
        st.text_input("Question", key="query_input", placeholder="Ask anything about your uploaded documents...", label_visibility="collapsed")
        if st.button("Search & Answer →", type="primary", use_container_width=True):
            question = st.session_state.query_input.strip()
            if not question:
                render_error("Question required", "Enter a question before searching your knowledge base.")
            else:
                try:
                    with st.status("Searching your knowledge base...", expanded=True) as status:
                        status.write("✓ Retrieving semantically relevant chunks")
                        result = st.session_state.rag_chain.answer(question)
                        status.write("✓ Generating a grounded answer")
                        status.update(label="Answer ready", state="complete", expanded=False)
                    st.session_state.last_result, st.session_state.last_question = result, question
                except Exception as exc:
                    render_error("Answer generation failed", "We could not complete this question. Confirm your API configuration and try again.", exc)
        if st.session_state.last_result is not None:
            render_result(st.session_state.last_result, st.session_state.last_question)
    st.markdown("### HOW NIYANTRA WORKS")
    steps = [
        ("01", "Upload", "Your documents are securely loaded."),
        ("02", "Chunk", "Documents are split into searchable pieces."),
        ("03", "Embed", "Chunks are converted into semantic vectors."),
        ("04", "Retrieve", "FAISS finds the most relevant context."),
        ("05", "Generate", "A concise answer is created using retrieved context only."),
        ("06", "Cite", "The answer is linked back to its source."),
    ]
    for row in (steps[:3], steps[3:]):
        for column, (number, title, copy) in zip(st.columns(3), row):
            with column:
                st.markdown(f'<div class="workflow-card"><div class="workflow-number">{number}</div><div class="workflow-title">{title}</div><div class="workflow-copy">{copy}</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
