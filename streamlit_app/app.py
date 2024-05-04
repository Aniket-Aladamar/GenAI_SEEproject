import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from io import BytesIO
import tempfile
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Load the models
llm = ChatGoogleGenerativeAI(model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Retrieval chain template
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""



def load_custom_css():
    custom_css = """
    <style>
    /* Add your custom CSS here */
    .chat-message {
        padding: 10px;
        border-radius: 25px;
    }
    .user-message {
        background-color: #f0f2f5;
        text-align: right;
    }
    .bot-message {
        background-color: #0099ff;
        color: white;
        text-align: left;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def process_pdf(in_memory_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        in_memory_file.seek(0)
        tmpfile.write(in_memory_file.read())
        tmpfilepath = tmpfile.name

    try:
        loader = PyPDFLoader(tmpfilepath)
        text_splitter = CharacterTextSplitter(separator=".", chunk_size=4000, chunk_overlap=0, length_function=len, is_separator_regex=False)
        pages = loader.load_and_split(text_splitter)
        vectordb = Chroma.from_documents(pages, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    finally:
        os.remove(tmpfilepath)
    return retriever



import streamlit as st
from PIL import Image

# Ensure you have the images named 'user.png' and 'bot.png' in the './images/' directory relative to your Streamlit script.
# Adjust the paths if your images are located in a different directory.

def chat_ui(retriever):
    st.write("😁😁Chat and Explore 😁😁")

    # Load images using PIL
    user_image = Image.open("streamlit_app/images/user.png")
    bot_image = Image.open("streamlit_app/images/bot.png")

    col1, col2 = st.columns([1, 5])
    with col1:
        # Display the user image
        st.image(user_image, width=50, caption='User')
    with col2:
        user_input = st.text_input("Ask a question about the PDF content:", "", key="user_input")

    if user_input:
        with st.spinner('Fetching response...'):
            try:
                prompt = PromptTemplate.from_template(template)
                # No change to your document chain creation as per your instruction
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
                response = retrieval_chain.invoke({"input": user_input})
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    # Display the bot image
                    st.image(bot_image, width=50, caption='Bot')
                with col2:
                    # Display the response
                    st.markdown(f"<div class='chat-message bot-message'>**Response:**\n\n{response['answer']}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f'Error getting response: {e}')


def main():
    load_custom_css()
    st.title("Retrival Augmented Generation BOT")
    st.write("Upload a PDF on RVU's timetable or document and explore😎😎")

    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    if uploaded_file is not None:
        with st.spinner('Processing upload...'):
            in_memory_file = BytesIO(uploaded_file.getvalue())
            st.success(f'File "{uploaded_file.name}" has been uploaded.')

            try:
                retriever = process_pdf(in_memory_file)
                st.success('File has been processed.')
                chat_ui(retriever)
            except Exception as e:
                st.error(f'Error processing file: {str(e)}')

if __name__ == "__main__":
    main()
