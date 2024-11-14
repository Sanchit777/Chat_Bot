from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
import re
import os
import uvicorn

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# API key for Google Generative AI
GOOGLE_API_KEY = 'AIzaSyC12OsKFkgv1RMg_xKSdKh1pE-fyMhbTsc'

# Helper function to extract text from PDFs
def get_pdf_text(pdf_paths: List[str]):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Split text into chunks for vector storage
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain with a custom prompt
def get_conversational_chain():
    prompt_template = """
    You are a knowledgeable Customer Care assistant. Your task is to provide an accurate answer to the given question and also what user wants. Follow these guidelines:

    Answer Precision: Ensure the answer is accurate.
    Contextual Awareness: If the answer is not available within the given context, respond with, "I am sorry, I can't assist you with that."
    User Interaction:
    If the user greets you, greet them back warmly and ask how you can assist them further.
    If the user requests elaboration, rephrase and explain the same information differently without introducing any inaccuracies.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3,
                                   google_api_key=GOOGLE_API_KEY)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Detect greetings in the user input
def detect_greeting(text):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    for greeting in greetings:
        if re.search(r'\b' + re.escape(greeting) + r'\b', text.lower()):
            return True
    return False

# Load PDF and prepare vector store on startup
@app.on_event("startup")
async def startup_event():
    pdf_paths = ["Guide.pdf"]  
    raw_text = get_pdf_text(pdf_paths)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    print("PDF processing complete and vector store created.")

# Define the request schema
class QuestionRequest(BaseModel):
    question: str

# Endpoint to handle questions
@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    question = request.question
    if detect_greeting(question):
        return {"answer": "Hello! How can I assist you today?"}
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    return {"answer": response["output_text"]}

# Run the FastAPI application on port 8080
if __name__ == "__main__":
    uvicorn.run("term:app", host="0.0.0.0", port=8080, reload=True)
