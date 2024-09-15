import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders   
 import   
 TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import   
 FAISS

# Set your OpenAI API key 
os.environ["OPENAI_API_KEY"] = "your_actual_api_key"

# Load your data
loaders = [
    TextLoader("path/to/your/text_file.txt"),
    PyPDFLoader("path/to/your/pdf_file.pdf")
]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)   


# Create   
 embeddings
embeddings = OpenAIEmbeddings()

# Store embeddings in a vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)

# Define your prompt template
template = """You are a helpful assistant providing information about Korean cuisine, specifically drawing upon the context provided below.

Context: {context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize the LLM
llm = OpenAI(temperature=0.7) 

# Create the chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True)   


# Get user input (if using Streamlit, use st.text_input)
question = input("Enter your question about Korean cuisine: ")

# Run the chain
result = qa_chain({"query": question})
answer, source_documents = result['result'], result['source_documents']

# Print or display the response
print(answer)

# (Optional) Print the source documents used for generating the answer
for source_document in source_documents:
    print("\nSource Document:")
    print(source_document.page_content)