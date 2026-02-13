import subprocess
import sys

def install_packages():
    """Install required packages for evaluation."""
    packages = [
        "langchain",
        "langchain-ollama",
        "langchain-chroma",
        "langchain-community",
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "nltk",
        "matplotlib",
        "seaborn",
        "chromadb",
        "pypdf",
        "tiktoken",
        "sentence-transformers"
    ]
    
    print("Installing required packages...")
    print("="*60)
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\n" + "="*60)
    print("Installation complete!")
    print("\nNext steps:")
    print("1. Make sure Ollama is running (ollama serve)")
    print("2. Pull your models: ollama pull llama2:latest mistral:latest olmo2:latest")
    print("3. Run vector.py to create your vector store")
    print("4. Create evaluation_dataset.json with your questions")
    print("5. Run evaluator.py to start the evaluation")

if __name__ == "__main__":
    install_packages()