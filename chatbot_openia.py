
"""

!pip install -q -U google-generativeai
!pip install -q langchain-google-genai
!pip install -q langchain
!pip install -q unstructured
!pip install -q chromadb
!pip install -q tiktoken
!pip install -U langchain-community
!pip install -q pypdf
!pip install -q python-dotenv
!pip install -q langchain_google_genai
"""



from langchain.document_loaders import PyPDFLoader  # Para cargar PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Para partir documentos
from dotenv import load_dotenv  # Para cargar variables de entorno
import os  # Para manejo de variables del sistema
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma  # Para crear el almacén de vectores

from langchain.chains import RetrievalQA  # Cadena de recuperación de preguntas y respuestas
from langchain_core.prompts import ChatPromptTemplate  # Prompt template correcto
from langchain.chains.combine_documents import create_stuff_documents_chain  # Para combinar documentos
from langchain.chains import create_retrieval_chain  # Cadena de recuperación
from langchain_core.messages import HumanMessage, AIMessage  # Para el historial de mensajes
from langchain.chains.history_aware_retriever import create_history_aware_retriever  # Recuperador basado en historial

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


def get_documents_from_pdf(pdf):
   
    #usar el loader con pypdf
    loader = PyPDFLoader(pdf)
    #Se carga el contenido del documento PDF utilizando el método load() del objeto loader
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    split_docs = text_splitter.split_documents(data)
    return split_docs


def create_db(docs):
    #crear los Embedding y vectorstore
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore

def create_chain(vectorstore):
    # Configura el modelo de lenguaje
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        api_key=api_key,
        temperature=0.4
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
        # retriever,
        history_aware_retriever,
        chain
    )

    return retrieval_chain



def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]

if __name__ == '__main__':
    
    # Obtener el api_key de Google Generative AI
    load_dotenv()
    api_key = os.getenv("OPENIA_API_KEY")
    
    if api_key is None:
        raise ValueError("No se encontró la API KEY en el archivo .env")

    # Ubicación del PDF
    pdf_loc = "de-la-brevedad-de-la-vida.pdf"
    
    # Procesar el PDF
    docs = get_documents_from_pdf(pdf_loc)
    
    # Crear vectorstore y la cadena de preguntas
    vectorstore = create_db(docs)
    chain = create_chain(vectorstore)

    chat_history = []

    while True:
        user_input = input("Tu: ")
        if user_input.lower() == 'exit':
            break

        # Procesar la entrada del usuario
        response = process_chat(chain, user_input, chat_history)
        
        # Actualizar historial de chat
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Asistente:", response)

  
