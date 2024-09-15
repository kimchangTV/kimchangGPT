import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set   
 your OpenAI API key (ensure it's kept secure!)
os.environ["OPENAI_API_KEY"] = "your_actual_api_key"

# Define your prompt template
template = """You are a helpful assistant providing information about Korean cuisine.

Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize the LLM
llm = OpenAI(temperature=0.7) # Adjust temperature for creativity

# Create the chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Get user input (if using Streamlit, use st.text_input)
question = input("Enter your question about Korean cuisine: ")

# Run the chain
response = llm_chain.run(question)

# Print or display the response
print(response) 