__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import streamlit as st
import os
import time
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")


# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] 

# Load your data
loader = TextLoader("data.txt")
docs = loader.load()

# Load PDF files
#pdf_loaders = [PyPDFLoader(f) for f in ["1.pdf"]]
#for pdf_loader in pdf_loaders:
#    docs.extend(pdf_loader.load_and_split())

# Load Excel files (assuming they have a 'content' column)
excel_loaders = [CSVLoader(f) for f in ["1-2.csv"]]  # Replace with your Excel file names
for excel_loader in excel_loaders:
    docs.extend(excel_loader.load())

# Specify the model name you're using 
#model_name = "jhgan/ko-sroberta-multitask" 

# Load the tokenizer associated with your model
#tokenizer = AutoTokenizer.from_pretrained(model_name) 

# Set the clean_up_tokenization_spaces parameter in the tokenizer
#tokenizer.clean_up_tokenization_spaces = False

# Create embeddings using the tokenizer
#embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"tokenizer": tokenizer}) 
#db = Chroma.from_documents(docs, embeddings)
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")  
db = Chroma.from_documents(docs, embeddings, collection_name="langchain", persist_directory="./")

# Set up LLM (ChatOpenAI) and RetrievalQA chain
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")  # or any other suitable chat model
retriever = db.as_retriever(search_kwargs={"k": 1})

# Construct the prompt templates
system_message_prompt = SystemMessagePromptTemplate.from_template(
    "당신은 한국어를 사용하는 유용한 챗봇입니다. 다음은 검색 결과를 바탕으로 사용자의 질문에 답변하는 데 사용할 수 있는 컨텍스트입니다. 컨텍스트와 함께 사용자의 질문에 답변하십시오. 컨텍스트가 답변에 충분하지 않은 경우 '잘 모르겠습니다.'라고 답변하십시오.\n\n{context}"
)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    chain_type_kwargs={"prompt": chat_prompt},
    input_key="question"  # Add this line
)

# Streamlit UI
st.title("DJ-GPT")

school_level = st.selectbox("학교 구분", ["초등학교", "중학교", "고등학교"])
grade = st.text_input("학년")
subject = st.text_input("과목")
num_classes = st.number_input("차시", min_value=1, value=1)

# Add input field for "성취 기준"
achievement_criteria = st.text_input("성취 기준", help="성취 기준, 또는 관련 키워드를 입력해주세요")

# Add a text box for additional information
additional_info = st.text_area("추가 정보 (선택 사항)")

if st.button("생성"):
    # Combine the base query with additional info and achievement criteria
    query = f"{school_level} {grade} 학생들을 위한 {subject} 과목과 관련된 {num_classes} 차시 동안 진행할 수 있는 과외 활동을 추천해주세요. {additional_info} "

    # If achievement criteria is provided, add it to the query
    if achievement_criteria:
        query += f"다음 성취 기준과 관련된 활동을 디자인해주세요: '{achievement_criteria}'"

    with st.spinner("답변 생성 중..."):
        result = qa_chain({"question": query})

        response_placeholder = st.empty()
        full_response = ""

        for chunk in result['result']:
            full_response += chunk
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
            time.sleep(0.05)

        # Attempt to identify relevant "성취 기준" based on keywords (refined logic)
        if achievement_criteria:
            keywords = achievement_criteria.split(",")

            matching_criteria = []
            for doc in docs:
                if any(keyword.strip() in doc.page_content for keyword in keywords):
                    # Extract the "성취 기준" phrase using regex
                    import re
                    matches = re.findall(r'\[.*?\].*', doc.page_content)
                    if matches:
                        matching_criteria.extend(matches)

            if matching_criteria:
                st.write("**관련 성취 기준:**")
                for criteria in matching_criteria:
                    st.write(criteria)
