# RAG-GEMINI-LangChain

`RAG-GEMINI-LangChain` is a Python-based project designed to integrate `Google's Generative AI` with `LangChain` for document understanding and information retrieval. This project enables users to ask questions about the content of PDF documents and receive accurate, context-aware answers. It utilizes Google Generative AI models along with LangChain's powerful document processing and retrieval capabilities.

## Getting Started

### Prerequisites

Ensure you have Python `3.12.2+` and pip installed on your system. This project also requires a virtual environment to manage dependencies.

### Installation

1. Clone the repository to your local machine:

```bash
https://github.com/NoManNayeem/RAG-GEMINI-LangChain.git
```
2. Navigate to the project directory:
```bash
cd RAG-GEMINI-LangChain
```

3. Create a virtual environment:
```bash
python -m venv venv
```

4. Activate the virtual environment:

- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```

- On macOS and Linux:
  ```bash
  source venv/bin/activate
  ```

5. Install the required dependencies:
```bash
pip install -r "Test Gemini RAG/requirements.txt" # for gemini.py

pip install -r "streamlit_app/requirements.txt" # for app.py
```

6. Set up your environment variables by creating a `.env` file in the root directory and adding your Google API Key and then create and activate virtual environemnt:
```bash
GOOGLE_API_KEY = Your_GOOGLE_API_KEY
```



### Running the Application

- To start the Streamlit application for interacting with PDFs:
```bash
streamlit run streamlit_app/app.py
```

- To run the example script for processing a specific `PDF` and querying it:
```bash
python Test Gemini RAG/gemini.py
```


## Features

- **PDF Processing:** Load and process PDF documents to make their content queryable.
- **Query Interface:** A Streamlit-based UI to ask questions and get answers from the processed PDF content.
- **Modular Structure:** Easy to extend or modify to integrate additional models or document types.

## Project Structure

- `streamlit_app/`: Contains the Streamlit application for interactive querying.
- `Test Gemini RAG/`: Example scripts and requirements for setting up and testing the integration.
- `.env`: Environment variables file.
- `venv/`: Virtual environment directory (should be created during setup).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements or suggestions.




