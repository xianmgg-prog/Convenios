import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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
                # Guardar el PDF temporalmente porque PyPDFLoader necesita un archivo físico
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # 1. Cargar PDF
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    # 2. Dividir texto (Chunking)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(docs)
                    
                    # 3. Crear Embeddings y Base de Datos Vectorial en memoria
                    embeddings = OpenAIEmbeddings(api_key=api_key)
                    vector_store = FAISS.from_documents(splits, embeddings)
                    
                    # Guardar en la sesión
                    st.session_state.vector_store = vector_store
                    st.success("¡Convenio procesado y listo para chatear!")
                    
                except Exception as e:
                    st.error(f"Error al procesar: {e}")
                finally:
                    # Limpiar archivo temporal
                    os.unlink(tmp_file_path)
    elif uploaded_file and not api_key:
        st.warning("Por favor, introduce tu clave API de OpenAI para continuar.")

# Mostrar el historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de chat
if prompt := st.chat_input("Ej: ¿Cuántos días de asuntos propios me corresponden?"):
    if not api_key:
        st.error("Introduce la API Key en la barra lateral primero.")
        st.stop()
        
    if st.session_state.vector_store is None:
        st.error("Sube y procesa un convenio PDF antes de preguntar.")
        st.stop()
        
    # Añadir pregunta del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el convenio..."):
            # Configurar el LLM
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
            
            # Crear el prompt estricto (Anti-alucinaciones)
            system_prompt = (
                "Eres un graduado social experto en derecho laboral español. "
                "Utiliza los siguientes fragmentos de un convenio colectivo para responder a la pregunta. "
                "Si la respuesta no está en los fragmentos proporcionados, di EXACTAMENTE: "
                "'No encuentro esta información en el convenio subido.' NO inventes información.\n\n"
                "Contexto:\n{context}"
            )
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Crear la cadena RAG
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Ejecutar consulta
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            
    # Añadir respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": answer})
