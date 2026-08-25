"""Chat RAG sobre un convenio colectivo en PDF con Streamlit y LangChain."""

import os
import tempfile

import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Mensajes y URLs
NO_INFO_MESSAGE = "No encuentro esta información en el convenio subido"
REGCON_SEARCH_URL = "https://expinterweb.mites.gob.es/regcon/pub/buscadorTextosEstatal?language=es"

# Prompt actualizado para que razone y entienda preguntas generales
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""Eres un graduado social experto en convenios colectivos.

Lee el siguiente contexto del convenio y responde a la pregunta del usuario.
Normas:
1. Basa tu respuesta ÚNICAMENTE en el contexto proporcionado.
2. Puedes resumir, agrupar información o explicarla con palabras sencillas, pero no inventes datos ni salarios que no aparezcan.
3. Si la pregunta es muy general (ej. "vacaciones"), haz un resumen de todo lo que encuentres en el contexto sobre ese tema.
4. Si la pregunta es sobre categorías, grupos profesionales o salarios, revisa las tablas incluidas en el contexto y enumera literalmente las filas recuperadas.
5. Si definitivamente el contexto no habla de lo que pide el usuario, responde exactamente: {NO_INFO_MESSAGE}

Contexto del convenio:
{{context}}

Pregunta: {{question}}

Respuesta:""",
)

def reset_document_state() -> None:
    """Elimina el índice y la conversación asociados al PDF anterior."""
    st.session_state.vectorstore = None
    st.session_state.pdf_name = None
    st.session_state.messages = []

def initialise_session_state() -> None:
    """Crea las variables de sesión necesarias la primera vez que se carga la app."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

@st.cache_resource(show_spinner=False)
def get_local_embeddings() -> HuggingFaceEmbeddings:
    """Carga una vez el modelo multilingüe que vectoriza el PDF localmente."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vectorstore(uploaded_pdf) -> FAISS:
    """Extrae el PDF, lo divide en fragmentos y devuelve un índice FAISS en memoria."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_pdf.getvalue())
            temp_path = temp_file.name

        documents = PyPDFLoader(temp_path).load()
        if not documents:
            raise ValueError("El PDF no contiene texto que se pueda procesar.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No se han podido crear fragmentos a partir del PDF.")

        # Vectorización local
        embeddings = get_local_embeddings()
        return FAISS.from_documents(chunks, embeddings)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def create_qa_chain(vectorstore: FAISS, gemini_api_key: str) -> RetrievalQA:
    """Crea la cadena RAG con la API clásica y compatible de RetrievalQA."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", # <-- CORREGIDO AQUÍ
        temperature=0,
        api_key=gemini_api_key,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # Búsqueda por similitud pura recuperando más contexto (15 fragmentos)
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15},
        ),
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )

def apply_custom_style() -> None:
    """Aplica una presentación visual cuidada sin dejar de usar Streamlit."""
    st.markdown(
        """
        <style>
            :root {
                --ink: #172033;
                --muted: #64748b;
                --surface: #ffffff;
                --line: #dbe4ef;
                --navy: #153b6d;
                --blue: #2563eb;
                --sky: #eef6ff;
                --mint: #eaf8f1;
            }

            .stApp {
                background:
                    radial-gradient(circle at 88% 4%, #dceeff 0, transparent 26rem),
                    linear-gradient(180deg, #f8fbff 0%, #f2f6fb 100%);
                color: var(--ink);
            }

            .block-container {
                max-width: 1080px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #102f57 0%, #153b6d 100%);
            }

            section[data-testid="stSidebar"] * {
                color: #f8fbff;
            }

            section[data-testid="stSidebar"] input {
                background: rgba(255, 255, 255, 0.96) !important;
                color: var(--ink) !important;
                border-radius: 10px !important;
            }

            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background: rgba(255, 255, 255, 0.10);
                border: 1px dashed rgba(255, 255, 255, 0.55);
                border-radius: 14px;
            }

            .sidebar-brand {
                padding: 0.25rem 0 0.8rem;
            }

            .sidebar-brand__eyebrow {
                color: #9fc6ff !important;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.11em;
                text-transform: uppercase;
            }

            .sidebar-brand h2 {
                color: #ffffff !important;
                font-size: 1.32rem;
                margin: 0.22rem 0 0;
            }

            .app-hero {
                background: linear-gradient(135deg, #ffffff 0%, #edf6ff 100%);
                border: 1px solid rgba(37, 99, 235, 0.14);
                border-radius: 22px;
                box-shadow: 0 12px 34px rgba(15, 47, 87, 0.08);
                margin-bottom: 1.5rem;
                overflow: hidden;
                padding: 2.1rem 2.3rem;
                position: relative;
            }

            .app-hero::after {
                background: #69a6ee;
                border-radius: 999px;
                content: "";
                height: 13rem;
                opacity: 0.14;
                position: absolute;
                right: -4rem;
                top: -7rem;
                width: 13rem;
            }

            .app-hero__eyebrow {
                color: var(--blue);
                font-size: 0.76rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .app-hero h1 {
                color: var(--ink);
                font-size: clamp(1.8rem, 4vw, 2.55rem);
                letter-spacing: -0.045em;
                line-height: 1.08;
                margin: 0.4rem 0 0.55rem;
            }

            .app-hero p {
                color: var(--muted);
                font-size: 1.02rem;
                margin: 0;
                max-width: 43rem;
            }

            [data-testid="stChatMessage"] {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--line);
                border-radius: 16px;
                box-shadow: 0 5px 17px rgba(15, 47, 87, 0.045);
                margin: 0.72rem 0;
                padding: 0.35rem 0.65rem;
            }

            [data-testid="stChatInput"] {
                background: var(--surface);
                border: 1px solid #c7d8ec;
                border-radius: 15px;
                box-shadow: 0 8px 24px rgba(15, 47, 87, 0.07);
            }

            [data-testid="stChatInput"]:focus-within {
                border-color: #5790dd;
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
            }

            [data-testid="stBaseButton-secondary"] {
                border-radius: 10px;
            }

            [data-testid="stAlert"] {
                border-radius: 12px;
            }

            @media (max-width: 640px) {
                .block-container { padding-top: 1rem; }
                .app-hero { border-radius: 17px; padding: 1.55rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    st.set_page_config(
        page_title="Consulta de convenios",
        page_icon="📄",
        layout="wide",
    )
    initialise_session_state()
    apply_custom_style()

    st.markdown(
        """
        <section class="app-hero">
            <div class="app-hero__eyebrow">Asistente laboral · RAG seguro</div>
            <h1>Consulta tu convenio colectivo</h1>
            <p>Sube un PDF y obtén respuestas basadas exclusivamente en su contenido.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand__eyebrow">Área laboral</div>
                <h2>Configuración</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gemini_api_key = st.text_input(
            "API Key de Gemini",
            type="password",
            help="Crea una clave en Google AI Studio. Se usa solo durante esta sesión.",
        )
        uploaded_pdf = st.file_uploader(
            "Sube un convenio en PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )
        st.divider()
        st.subheader("Buscador de convenios")
        st.caption(
            "Encuentra el texto oficial en REGCON y vuelve aquí para subir el PDF."
        )
        st.link_button(
            "Abrir buscador oficial REGCON ↗",
            REGCON_SEARCH_URL,
            use_container_width=True,
        )
        st.caption("El PDF se vectoriza localmente; Gemini solo responde al chat.")

    if uploaded_pdf and uploaded_pdf.name != st.session_state.pdf_name:
        reset_document_state()

    if uploaded_pdf and st.session_state.vectorstore is None:
        try:
            with st.spinner("Leyendo y preparando el convenio..."):
                st.session_state.vectorstore = build_vectorstore(uploaded_pdf)
                st.session_state.pdf_name = uploaded_pdf.name
            st.success(f"Convenio preparado: {uploaded_pdf.name}")
        except Exception as error:
            st.error(f"No se ha podido procesar el PDF: {error}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Pregunta sobre el convenio",
        disabled=st.session_state.vectorstore is None or not gemini_api_key,
    )

    if st.session_state.vectorstore is not None and not gemini_api_key:
        st.info("Introduce tu API Key de Gemini para hacer preguntas al convenio.")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando en el convenio..."):
                    qa_chain = create_qa_chain(
                        st.session_state.vectorstore,
                        gemini_api_key,
                    )
                    response = qa_chain.invoke({"query": question})
                    answer = response["result"]
                st.markdown(answer)
                source_pages = sorted(
                    {
                        document.metadata.get("page", 0) + 1
                        for document in response.get("source_documents", [])
                    }
                )
                if source_pages:
                    pages = ", ".join(str(page) for page in source_pages)
                    st.caption(f"Fragmentos consultados: páginas {pages}")
            except Exception as error:
                answer = f"No se ha podido obtener una respuesta: {error}"
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
