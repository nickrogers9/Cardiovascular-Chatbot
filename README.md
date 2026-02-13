# Cardiovascular Disease Virtual Assistant

This project implements a chatbot that answers questions about cardiovascular diseases using locally-run Large Language Models (LLMs). It was developed as a Bachelor's thesis at Vietnamese-German University.

## Features
- Three LLMs: Llama2 7B, Mistral 7B, OLMo2 7B (switchable via UI)
- Retrieval-Augmented Generation (RAG) using PDF documents
- Streamlit web interface
- Local execution (no cloud, privacy-preserving)
- Evaluation framework to compare model performance

## Project Structure

.

├── app.py # Streamlit web application

├── vector.py # Document processing & vector store creation

├── evaluator.py # Model evaluation script

├── install_requirements.py # Package installer

├── run_evaluation.bat # Windows batch file to run evaluation

├── evaluation_dataset.json # 18 questions with ground truth for evaluation

├── cardiovascular_docs/ # Folder with source PDF documents

├── chroma_db_optimized/ # Vector database (created by vector.py, excluded from git)

└── README.md # This file


## Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM, 4GB GPU VRAM recommended (but CPU-only works)
- The following Ollama models pulled:

bash:

  ollama pull llama2:latest
  
  ollama pull mistral:latest
  
  ollama pull olmo2:latest
  
  ollama pull nomic-embed-text

## Setup & Installation

1. Clone the repository (if you downloaded as ZIP, extract it):

bash:

  git clone https://github.com/nickrogers9/Cardiovascular-Chatbot.git
  
  cd Cardiovascular-Chatbot

2. Create a virtual environment (recommended):

bash:

  python -m venv .venv
  
  .venv\Scripts\activate   # on Windows

3. Install dependencies:

bash:

  python install_requirements.py

Or manually install:

bash:

  pip install langchain langchain-ollama langchain-chroma streamlit pandas numpy scikit-learn nltk matplotlib seaborn pypdf

4. Build the vector store:

bash:

  python vector.py

5. Run the chatbot:

bash:

  streamlit run app.py

The web interface will open in your browser.

## Usage

- Type your question in the chat box.
- Select a model from the sidebar (Llama2, Mistral, OLMo2).
- The assistant will retrieve relevant document chunks and generate an answer based only on those sources.
- Expand "View Sources Used" to see which parts of the documents were referenced.

## Evaluation

To reproduce the evaluation results from the thesis:

bash:

  python evaluator.py

Or double-click run_evaluation.bat (Windows only). This will run all 18 questions through each model and generate performance reports (CSV, JSON, plots).

## Notes

- All processing is done locally; no data leaves your machine.
- The system is not a medical diagnostic tool – always consult a healthcare professional.
- If you encounter errors, ensure Ollama is running (ollama serve in a separate terminal).
