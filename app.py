from flask import Flask, request, jsonify, render_template
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Flask app
app = Flask(__name__)


# Ruta fija para el PDF
PDF_PATH = "de-la-brevedad-de-la-vida.pdf"

# Cargar API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("No se encontró la API KEY en el archivo .env")

# 1. Procesar el PDF fijo
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = text_splitter.split_documents(data)
    return split_docs

# 2. Crear el vectorstore usando el PDF procesado
def create_db(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore

# 3. Crear la cadena de preguntas y respuestas
def create_chain(vectorstore):
    llm = GoogleGenerativeAI(
        model='gemini-1.5-flash',
        google_api_key=api_key,
        temperature=0.4,
        language="spanish"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions in spanish based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

# 4. Procesar la conversación
def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]

# Procesar el PDF una vez al inicio
docs = process_pdf(PDF_PATH)
vectorstore = create_db(docs)
chain = create_chain(vectorstore)
chat_history = []

#Pagina principal del formulario
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para recibir preguntas
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    
    # Procesar la pregunta
    response = process_chat(chain, question, chat_history)
    
    # Añadir la pregunta y la respuesta al historial del chat
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)