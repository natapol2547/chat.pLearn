# chat.pLearn

### Code for converting PDF files to Vector store and store it locally [FAISS]

```python
from langchain.document_loaders import TextLoader

import os
os.environ["OPENAI_API_KEY"] = "REPLACE-WITH-OPENAI-KEY"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
pip install pypdf

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("REPLACE-WITH-PATH-TO-PDF")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)

db.save_local("REPLACE-WITH-A-NAME")
```

### Loading Vector store and do Q&A on the PDF using OpenAI LLM (Also return Source Documents)

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

os.environ["OPENAI_API_KEY"] = "REPLACE-WITH-OPENAI-KEY"

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("REPLACE-WITH-A-NAME", embeddings)

from langchain.chat_models import ChatOpenAI

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    db.as_retriever(),
    condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
    return_source_documents=True
)
chat_history = []
while True:
  query = input("Input your question: ")
  result = qa({"question": query, "chat_history": chat_history})
  print(result['source_documents'])
  chat_history.append((query, result["answer"]))
```
