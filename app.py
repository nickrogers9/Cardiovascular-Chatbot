import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import subprocess
import time
import os

# 1. Set up page
st.set_page_config(
    page_title="Cardio Assistant",
    page_icon="❤️",
    layout="wide"
)

# 2. Load the vector store (with caching to avoid reloading)
@st.cache_resource
def load_retriever():
    """Load the existing vector store from disk"""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        db_location = "./chroma_db_optimized"
        
        # Check if vector store exists
        if not os.path.exists(db_location):
            st.error(f"Vector store not found at {db_location}. Please run vector.py first.")
            return None
        
        vector_store = Chroma(
            persist_directory=db_location,
            embedding_function=embeddings,
            collection_name="medical_documents"
        )
        
        # Test the retriever
        test_results = vector_store.as_retriever(search_kwargs={"k": 3}).invoke("test")
        print(f"Vector store loaded successfully. Contains documents: {len(test_results) > 0}")
        
        return vector_store.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        print(f"Error details: {e}")
        return None

# 3. Initialize retriever
retriever = load_retriever()

# 4. Initialize session state (memory for the app)
if 'model' not in st.session_state:
    st.session_state.model = "llama2:latest"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 5. Define the prompt template
template = """
You are a medical AI assistant specializing in cardiovascular diseases.

IMPORTANT: You MUST base your answers ONLY on the provided context from medical documents.
If the context doesn't contain relevant information, say "Based on the provided medical documents, I don't have specific information about that."

Context from medical documents:
{context}

Question: {question}

Provide a clear, detailed answer based only on the context above. Do not make up information.
Include relevant statistics or findings when available.

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 6. Function to get model
def get_model(model_name="llama2:latest"):
    try:
        return OllamaLLM(model=model_name, base_url="http://localhost:11434", temperature=0.1)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None

# 7. Function to get answer
def get_answer(question, model_name):
    """Get answer using RAG"""
    if retriever is None:
        return "Vector store not loaded. Please run vector.py first to process documents.", []
    
    try:
        # Get relevant context
        context_docs = retriever.invoke(question)
        
        if not context_docs:
            return "No relevant documents found in the knowledge base. Please try a different question or add more documents.", []
        
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Get model
        model = get_model(model_name)
        if model is None:
            return f"Failed to load model {model_name}. Please check if it's available in Ollama.", []
        
        # Generate response
        chain = prompt | model
        result = chain.invoke({"context": context_text, "question": question})
        
        return result, context_docs
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        print(f"Error details: {e}")
        return f"An error occurred: {str(e)}", []

# 8. Sidebar for controls
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.subheader("Select Model")
    model_option = st.selectbox(
        "Choose AI Model",
        ["llama2:latest", "mistral:latest", "olmo2:latest"],
        index=0
    )
    
    if st.button("Apply Model"):
        st.session_state.model = model_option
        with st.spinner(f"Loading {model_option}..."):
            try:
                # Try to pull the model if not available
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if model_option not in result.stdout:
                    st.info(f"Pulling {model_option}...")
                    subprocess.run(["ollama", "pull", model_option], check=True, timeout=120)
                    st.success(f"Loaded {model_option}")
                else:
                    st.success(f"Using existing {model_option}")
            except subprocess.CalledProcessError as e:
                st.warning(f"Could not pull model automatically. Please run 'ollama pull {model_option}' manually.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        time.sleep(1)
        st.rerun()
    
    st.divider()
    
    # Show vector store status
    st.subheader("📚 Knowledge Base Status")
    if retriever:
        st.success("✓ Vector store loaded")
    else:
        st.error("✗ Vector store not loaded")
    
    st.markdown("To rebuild knowledge base:")
    if st.button("🔄 Re-run vector.py"):
        with st.spinner("Processing documents..."):
            try:
                result = subprocess.run(["python", "vector.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Documents processed successfully!")
                    # Clear cache to force reload
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"Error: {result.stderr}")
            except Exception as e:
                st.error(f"Failed to run vector.py: {str(e)}")
    
    st.divider()
    
    # Show recent chat history
    st.subheader("Recent Questions")
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history[-5:]):
            st.caption(f"Q{i+1}: {q[:50]}...")
    else:
        st.caption("No questions yet")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# 9. Main chat area
st.title("❤️ Cardiovascular Disease Medical Assistant")
st.markdown("Ask questions about cardiovascular diseases based on your uploaded documents")

# Display chat history
if st.session_state.chat_history:
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
        st.divider()
else:
    st.info("💡 Try asking questions like: 'What is hypertension?' or 'What are the symptoms of heart disease?'")

# 10. Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        if retriever is None:
            st.error("⚠️ Knowledge base not loaded. Please run vector.py first to process your documents.")
        else:
            with st.spinner("Searching knowledge base..."):
                answer, sources = get_answer(user_input, st.session_state.model)
                st.write(answer)
                
                # Show sources in expander
                if sources:
                    with st.expander("📚 View Sources Used"):
                        for i, doc in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            st.text(f"File: {source_file}")
                            st.text(f"Page: {doc.metadata.get('page', 'N/A')}")
                            st.text(f"Content: {doc.page_content[:200]}...")
                            st.divider()
                else:
                    st.info("No specific sources were retrieved for this question.")
    
    # Save to history
    st.session_state.chat_history.append((user_input, answer))