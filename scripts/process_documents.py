import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import openai
import numpy as np
import faiss
import json

# Carregar variáveis de ambiente
load_dotenv()

# Cliente OpenAI
client = openai.OpenAI()

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Função para extrair texto do TXT
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

# Função para dividir texto em chunks com metadados
def split_text_with_metadata(text, source, chunk_size=1000, chunk_overlap=200):
    # Usar regex para identificar artigos
    articles = re.split(r'(Art\. \d+)', text)
    documents = []
    current_article = ""
    prev_article = "Introdução"  # Inicializar
    for i, part in enumerate(articles):
        if re.match(r'Art\. \d+', part):
            if current_article:
                # Adicionar documento anterior
                doc = Document(page_content=current_article.strip(), metadata={"source": source, "article": prev_article})
                documents.append(doc)
            prev_article = part
            current_article = part
        else:
            current_article += part
    if current_article:
        doc = Document(page_content=current_article.strip(), metadata={"source": source, "article": prev_article})
        documents.append(doc)
    
    # Se não encontrou artigos, dividir genericamente
    if not documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = splitter.create_documents([text], metadatas=[{"source": source}])
        documents = docs
    
    return documents

# Caminhos
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
pdf_path = os.path.join(data_dir, 'Manual_Operacional_do_RGC_Consolidado_revisao_3.pdf')
txt_path = os.path.join(data_dir, 'Resolução Anatel nº 765_2023.txt')

# Extrair textos
pdf_text = extract_text_from_pdf(pdf_path)
txt_text = extract_text_from_txt(txt_path)

# Dividir em documentos
pdf_docs = split_text_with_metadata(pdf_text, "Manual Operacional do RGC")
txt_docs = split_text_with_metadata(txt_text, "Resolução Anatel nº 765/2023")

all_docs = pdf_docs + txt_docs

# Extrair textos e metadatas
texts = [doc.page_content for doc in all_docs]
metadatas = [doc.metadata for doc in all_docs]

# Gerar embeddings
embeddings_list = []
for text in texts:
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    embeddings_list.append(response.data[0].embedding)

embeddings_np = np.array(embeddings_list)

# Criar índice FAISS
index = faiss.IndexFlatL2(1536)  # Dimensão do ada-002
index.add(embeddings_np)

# Salvar índice
faiss.write_index(index, os.path.join(data_dir, 'faiss_index.idx'))

# Salvar textos e metadatas
with open(os.path.join(data_dir, 'texts.json'), 'w') as f:
    json.dump(texts, f)
with open(os.path.join(data_dir, 'metadatas.json'), 'w') as f:
    json.dump(metadatas, f)

print("Indexação concluída!")