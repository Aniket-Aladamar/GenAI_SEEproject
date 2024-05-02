### Install Tools
#pip install pypdf2
#pip install chromadb
#pip install google.generativeai
#pip install langchain-google-genai
#pip install langchain
#pip install langchain_community
#pip install jupyter

# Import Tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma



# Import GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")




# Load the models
llm = ChatGoogleGenerativeAI(model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



# Load the PDF and create chunks
loader = PyPDFLoader("./data/Tech_Lead_Resume (1).pdf")
text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
pages = loader.load_and_split(text_splitter)



# Turn the chunks into embeddings and store them in Chroma
vectordb=Chroma.from_documents(pages,embeddings)



# Configure Chroma as a retriever with top_k=5
retriever = vectordb.as_retriever(search_kwargs={"k": 5})


# Create the retrieval chain
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""

prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Invoke the retrieval chain
response=retrieval_chain.invoke({"input":"Tell me about Nayeem's experiences?"})

# Print the answer to the question
print(response["answer"])
