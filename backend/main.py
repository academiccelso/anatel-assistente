from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar vetor store
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
vectorstore_path = os.path.join(data_dir, 'faiss_index')
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

# Modelo de linguagem
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

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
        result = qa_chain.run(q.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Assistente da Anatel"} 