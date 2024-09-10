
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
# PARTE 1: CONFIGURAR LA DATA, SPLITTER, EMMBEDING Y MODELO

#1: cargar el pdf


from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
#usar el loader con pypdf
loader = PyPDFLoader("de-la-brevedad-de-la-vida.pdf")
#Se carga el contenido del documento PDF utilizando el método load() del objeto loader
data = loader.load()

# 2: partir la data para que pueda entrar en embeddings"""

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


# 3: obtener el api de gemini

from dotenv import load_dotenv
import os
#import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configura el modelo de lenguaje
llm = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key=api_key,
    temperature=0.4,
    language="spanish"
)


# 4. Embedding y vectorstore

from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

"""## PARTE 2. preguntarle cosas al chat sobre el pdf"""

from langchain import PromptTemplate
from langchain.chains import RetrievalQA


# Crear una plantilla de mensaje para respuestas en español
prompt_template = PromptTemplate(
    input_variables=["context"],
    template="You are a helpful assistant. Answer all questions to the best of your ability in spanish. Context: {context}"
)
# Crear una cadena de recuperación de preguntas y respuestas con la cadena LLM
cadena_resuelve_preguntas = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={'prompt': prompt_template}
)

# Ejemplo de uso:
pregunta = input("  Pon tu pregunta a chat")
respuesta = cadena_resuelve_preguntas.invoke(pregunta)
#print(respuesta["answer"])
print(respuesta)