"""Chat RAG sobre un convenio colectivo en PDF con Streamlit y LangChain."""

import os
import tempfile

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# La respuesta alternativa se mantiene exactamente como solicita el requisito.
NO_INFO_MESSAGE = "No encuentro esta información en el convenio subido"

# El modelo solo recibe los fragmentos recuperados en la variable {context}.
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""Eres un graduado social experto que asesora sobre convenios colectivos.

Responde la pregunta usando ÚNICAMENTE la información incluida en el contexto.
No uses conocimientos externos, no supongas datos, no completes lagunas y no inventes.
Si el contexto no contiene una respuesta clara y suficiente, responde exactamente:
{NO_INFO_MESSAGE}

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


def build_vectorstore(uploaded_pdf, api_key: str) -> FAISS:
    """Extrae el PDF, lo divide en fragmentos y devuelve un índice FAISS en memoria."""
    # NamedTemporaryFile con delete=False permite que PyPDFLoader abra el archivo en Windows.
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

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        return FAISS.from_documents(chunks, embeddings)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def create_qa_chain(vectorstore: FAISS, api_key: str) -> RetrievalQA:
    """Crea la cadena RAG con la API clásica y compatible de RetrievalQA."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )


def main() -> None:
    st.set_page_config(page_title="Consulta de convenios", page_icon="📄")
    initialise_session_state()

    st.title("📄 Consulta tu convenio colectivo")
    st.caption("Las respuestas se basan exclusivamente en el PDF subido.")

    with st.sidebar:
        st.header("Configuración")
        api_key = st.text_input(
            "API Key de OpenAI",
            type="password",
            help="La clave se usa solo durante esta sesión.",
        )
        uploaded_pdf = st.file_uploader(
            "Sube un convenio en PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )

    # Al elegir otro PDF se descarta el índice y el historial anteriores.
    if uploaded_pdf and uploaded_pdf.name != st.session_state.pdf_name:
        reset_document_state()

    if uploaded_pdf and st.session_state.vectorstore is None:
        if not api_key:
            st.info("Introduce tu API Key de OpenAI para procesar el convenio.")
        else:
            try:
                with st.spinner("Leyendo y preparando el convenio..."):
                    st.session_state.vectorstore = build_vectorstore(uploaded_pdf, api_key)
                    st.session_state.pdf_name = uploaded_pdf.name
                st.success(f"Convenio preparado: {uploaded_pdf.name}")
            except Exception as error:
                st.error(f"No se ha podido procesar el PDF: {error}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Pregunta sobre el convenio",
        disabled=st.session_state.vectorstore is None,
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando en el convenio..."):
                    qa_chain = create_qa_chain(st.session_state.vectorstore, api_key)
                    # El nombre de entrada de RetrievalQA es "query".
                    response = qa_chain.invoke({"query": question})
                    answer = response["result"]
                st.markdown(answer)
            except Exception as error:
                answer = f"No se ha podido obtener una respuesta: {error}"
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
