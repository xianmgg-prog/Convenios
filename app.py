import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuración de la página
st.set_page_config(page_title="Chat Convenios Laborales", page_icon="⚖️", layout="wide")

st.title("⚖️ Asistente de Convenios Colectivos")
st.markdown("Sube un convenio en PDF y hazle preguntas. El asistente basará sus respuestas **estrictamente** en el documento.")

# Inicializar historial de chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Barra lateral para configuración y carga de archivos
with st.sidebar:
    st.header("1. Configuración")
    api_key = st.text_input("Clave API de OpenAI:", type="password", help="Tu clave no se guarda, se usa solo en esta sesión.")
    
    st.header("2. Subir Convenio")
    uploaded_file = st.file_uploader("Sube el PDF del convenio", type=["pdf"])
    
    if uploaded_file and api_key:
        if st.button("Procesar PDF"):
            with st.spinner("Procesando y analizando el convenio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
                    
                    embeddings = OpenAIEmbeddings(api_key=api_key)
                    vector_store = FAISS.from_documents(splits, embeddings)
                    
                    st.session_state.vector_store = vector_store
                    st.success("¡Convenio procesado y listo para chatear!")
                    
                except Exception as e:
                    st.error(f"Error al procesar: {e}")
                finally:
                    os.unlink(tmp_file_path)
    elif uploaded_file and not api_key:
        st.warning("Por favor, introduce tu clave API de OpenAI para continuar.")

# Mostrar el historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de chat
if user_question := st.chat_input("Ej: ¿Cuántos días de asuntos propios me corresponden?"):
    if not api_key:
        st.error("Introduce la API Key en la barra lateral primero.")
        st.stop()
        
    if st.session_state.vector_store is None:
        st.error("Sube y procesa un convenio PDF antes de preguntar.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
        
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el convenio..."):
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
            
            system_prompt = """Eres un graduado social experto en derecho laboral español.
            Utiliza los siguientes fragmentos de un convenio colectivo para responder a la pregunta.
            Si la respuesta no está en los fragmentos proporcionados, di EXACTAMENTE: 'No encuentro esta información en el convenio subido.' NO inventes información.

            Contexto:
            {context}

            Pregunta: {question}
            Respuesta:"""
            
            PROMPT = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
            
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            response = qa_chain.invoke({"query": user_question})
            answer = response["result"]
            
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
