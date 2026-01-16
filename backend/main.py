from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv
import openai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Carregar variáveis de ambiente
load_dotenv()

# Cliente OpenAI
client = openai.OpenAI()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar índice FAISS e dados
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
index_path = os.path.join(data_dir, 'faiss_index.idx')
texts_path = os.path.join(data_dir, 'texts.json')
metadatas_path = os.path.join(data_dir, 'metadatas.json')

index = faiss.read_index(index_path)
with open(texts_path, 'r') as f:
    texts = json.load(f)
with open(metadatas_path, 'r') as f:
    metadatas = json.load(f)

# Modelo de linguagem
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Função para buscar contexto
def search_context(question, k=5):
    # Gerar embedding da pergunta
    response = client.embeddings.create(input=question, model="text-embedding-ada-002")
    query_emb = np.array([response.data[0].embedding])
    
    # Buscar no índice
    D, I = index.search(query_emb, k)
    
    # Recuperar textos e metadatas
    contexts = []
    for i in I[0]:
        context = texts[i]
        metadata = metadatas[i]
        contexts.append(f"Documento: {metadata['source']}, Artigo: {metadata.get('article', 'N/A')}\n{context}")
    
    return "\n\n".join(contexts)

# Prompt de sistema
system_prompt = """
Você é um assistente informativo da Anatel.
Você só pode responder usando o conteúdo fornecido nos documentos oficiais incluídos no contexto.
Se a pergunta não puder ser respondida com base nesses documentos, informe claramente que a informação não está disponível.
Use linguagem simples, clara e acessível ao consumidor.
Sempre cite o dispositivo normativo que fundamenta a resposta.
"""

prompt_template = system_prompt + "\n\nContexto:\n{context}\n\nPergunta: {question}\n\nResposta:"

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT}
)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    try:
        context = search_context(q.question)
        prompt = PROMPT.format(context=context, question=q.question)
        response = llm(prompt)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Assistente da Anatel"} 