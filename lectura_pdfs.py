# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.

#!pip install langchain unstructured[pdf] pypdf chromadb openai tiktoken


#!pip install -U langchain-community


from langchain.document_loaders import UnstructuredFileLoader
#sirve para pdfs solamente
from langchain.document_loaders import PyPDFLoader
#usar el loader con pypdf
loader = PyPDFLoader("/content/INTELIGENCIA ARTIFICIAL Y PENSAMIENTO CRÍTICO.pdf")
#Se carga el contenido del documento PDF utilizando el método load() del objeto loader
data = loader.load()

"""para cargar multiples pdfs, crear una carpeta en content con los  pdfs la carpeta puede tener cualquier nombre como "PDFS"y luego usar el codigo:
loader=PyPDFDirectoryLoader("PDFS")
data=loader.load()"""

#obtener el api

key = "xxxxxxxxxx"

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI


# Los documentos NO pueden ser procesados directamente por LLMs porque contienen demasiado texto, sin embargo, se puede
# particionarlo en conjuntos de texto más pequeños para entonces poder acceder a su información.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cada partición contendrá 1500 palabras, y tendrán una intersección de 200, de modo que la cadena 2 comparte 200
# palabras con la cadena 1 y con la cadena 3
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
    )

documents = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(api_key=key)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)


#se carga la libreria para cargar el modelo de openia dentro de lagnchain
from langchain.chat_models import ChatOpenAI
# Inicializa el modelo de chat
chat_model= ChatOpenAI(
    model_name="gpt-3.5-turbo",
    api_key=key,
    n=1,
    temperature=0.3
)


##2. preguntarle cosas al chat sobre el pdf


#Este código utiliza la librería langchain para crear un modelo de recuperación de preguntas y respuestas (RetrievalQA) y luego ejecutar una consulta utilizando ese modelo.


from langchain.chains import RetrievalQA

#utiliza RetrievalQA.from_chain_type() para crear un modelo de recuperación de preguntas y respuestas
cadena_resuelve_preguntas=RetrievalQA.from_chain_type(
    llm=chat_model,
    #stuff implica que hará el analisis solo con lo que le quepa en el prompt
    chain_type="stuff",
    #retriever: Un objeto que actúa como recuperador de información, creado a partir de un vectorstore mediante as_retriever()
    retriever=vectorstore.as_retriever(search_kwargs={"k":2},)
)

cadena_resuelve_preguntas.run("con que esta mas vinculada la conexion del ser humano y la informacion disponible en internet")

cadena_resuelve_preguntas.run("que permite la recoleccion y analisis de datos")