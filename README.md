# Assistente do Consumidor da Anatel

Este é um aplicativo web simples que responde dúvidas de consumidores de telecomunicações com base na Resolução Anatel nº 765/2023 e no Manual Operacional do RGC.

## Arquitetura

- **Backend**: FastAPI (Python) com RAG usando LangChain, FAISS e OpenAI.
- **Frontend**: HTML/CSS/JS puro.
- **Base de Conhecimento**: Embeddings dos documentos oficiais.

## Como Rodar

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

2. Configure a chave da OpenAI:
   - Copie `.env.example` para `.env`.
   - Adicione sua chave da OpenAI no arquivo `.env`:
     ```
     OPENAI_API_KEY=sk-your-actual-key-here
     ```
   - Alternativamente, defina a variável de ambiente no terminal:
     ```
     export OPENAI_API_KEY=sk-your-actual-key-here
     ```

3. Execute o script de indexação:
   ```
   python scripts/process_documents.py
   ```

4. Inicie o backend:
   ```
   cd backend
   uvicorn main:app --reload
   ```

5. Abra o frontend: Abra `frontend/index.html` no navegador ou sirva com um servidor local.

## Limitações do MVP

- Respostas baseadas apenas nos documentos fornecidos.
- Não armazena dados pessoais.
- Requer chave da OpenAI.
- Interface simples, sem autenticação.

## Exemplos de Perguntas

- "Quais são os direitos do consumidor em caso de interrupção do serviço?"
- "Como solicitar ressarcimento por falha na prestação de serviço?"

Respostas incluirão citações como "Base legal: RGC – art. 70".

## Documentação Técnica

- Prompt de sistema: Restritivo, obriga citações e nega respostas fora do contexto.
- Fragmentação: Por artigos onde possível, senão por chunks de texto.
- Recuperação: Top-5 trechos mais relevantes.