from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import hashlib
import json
from pathlib import Path
from typing import List, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class OptimizedVectorStore:
    def __init__(self, db_path: str = "./chroma_db", embedding_model: str = "nomic-embed-text:latest"):
        """
        Initialize with optimized settings.
        Note: 'nomic-embed-text' is better for long documents despite being slower.
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.collection_name = "medical_documents"  # Consistent collection name

        # Initialize embeddings with GPU support if available
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434",
            num_gpu=1  # Use GPU if available
        )

        self.processed_files_path = os.path.join(db_path, "processed_files.json")
        self.processed_files = self._load_processed_files()
        
        # Optimized text splitter with larger chunks for better context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for better context
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
    def _load_processed_files(self) -> dict:
        """Load cache of already processed files"""
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_processed_files(self):
        """Save cache of processed files"""
        os.makedirs(os.path.dirname(self.processed_files_path), exist_ok=True)
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed_files, f)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file to detect changes"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _load_pdf_with_progress(self, file_path: str) -> List:
        """Load PDF with progress tracking"""
        try:
            start_time = time.time()
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            elapsed = time.time() - start_time
            print(f"  Loaded {len(docs)} pages from {os.path.basename(file_path)} in {elapsed:.1f}s")
            return docs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def get_new_or_modified_files(self, folder_path: str) -> List[str]:
        """Only return files that are new or modified"""
        new_files = []
        
        # Support multiple file types
        extensions = ['.pdf', '.txt', '.csv', '.md']
        
        for ext in extensions:
            for file_path in Path(folder_path).rglob(f"*{ext}"):
                file_path_str = str(file_path)
                file_hash = self._get_file_hash(file_path_str)
                
                if not file_hash:  # Skip unreadable files
                    continue
                    
                # Check if file is new or modified
                if (file_path_str not in self.processed_files or 
                    self.processed_files[file_path_str] != file_hash):
                    new_files.append(file_path_str)
                    
        return new_files
    
    def process_folder(self, folder_path: str, force_reprocess: bool = False, 
                      batch_size: int = 50, max_workers: int = 2):
        """
        Process folder incrementally with batching and parallel loading
        """
        if force_reprocess:
            self.processed_files = {}
        
        # Get files that need processing
        files_to_process = (self.get_new_or_modified_files(folder_path) 
                           if not force_reprocess 
                           else [str(p) for p in Path(folder_path).rglob("*.*") 
                                 if p.suffix.lower() in ['.pdf', '.txt', '.csv', '.md']])
        
        if not files_to_process:
            print("No new or modified files to process.")
            return self._get_existing_retriever()
        
        print(f"Processing {len(files_to_process)} new/modified files...")
        
        # Load documents in parallel
        all_docs = []
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self._load_file, file_path): file_path 
                                 for file_path in files_to_process}
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        docs = future.result()
                        if docs:
                            all_docs.extend(docs)
                            # Update cache immediately
                            self.processed_files[file_path] = self._get_file_hash(file_path)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        else:
            for file_path in files_to_process:
                docs = self._load_file(file_path)
                if docs:
                    all_docs.extend(docs)
                    self.processed_files[file_path] = self._get_file_hash(file_path)
        
        if not all_docs:
            print("No documents loaded.")
            return self._get_existing_retriever()
        
        print(f"Loaded {len(all_docs)} total documents")
        
        # Split into chunks
        split_docs = self.text_splitter.split_documents(all_docs)
        print(f"Split into {len(split_docs)} chunks")
        
        # Process in batches to avoid embedding too much at once
        print(f"Adding to vector store in batches of {batch_size}...")
        
        # Check if vector store exists
        if os.path.exists(self.db_path):
            print("Adding to existing vector store...")
            vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Add in batches
            for i in range(0, len(split_docs), batch_size):
                batch = split_docs[i:i+batch_size]
                try:
                    vector_store.add_documents(batch)
                    print(f"  Added batch {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1} "
                          f"({len(batch)} documents)")
                except Exception as e:
                    print(f"Error adding batch {i//batch_size + 1}: {e}")
                    # Try smaller batch if error occurs
                    if batch_size > 10:
                        print("Retrying with smaller batch size...")
                        for j in range(0, len(batch), batch_size//2):
                            small_batch = batch[j:j+batch_size//2]
                            try:
                                vector_store.add_documents(small_batch)
                            except:
                                pass
        else:
            print("Creating new vector store...")
            vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                collection_name=self.collection_name
            )
        
        # Save processed files cache
        self._save_processed_files()
        
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def _load_file(self, file_path: str) -> List:
        """Load a single file based on extension"""
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith('.pdf'):
            return self._load_pdf_with_progress(file_path)
        elif file_path_lower.endswith('.txt'):
            try:
                loader = TextLoader(file_path)
                return loader.load()
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return []
        elif file_path_lower.endswith('.csv'):
            try:
                loader = CSVLoader(file_path)
                return loader.load()
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return []
        return []
    
    def _get_existing_retriever(self):
        """Get retriever from existing DB"""
        if os.path.exists(self.db_path):
            try:
                vector_store = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                return vector_store.as_retriever(search_kwargs={"k": 5})
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return None
        return None
    
    def query(self, question: str, retriever=None):
        """Simple query interface"""
        if retriever is None:
            retriever = self._get_existing_retriever()
        
        if retriever:
            return retriever.invoke(question)
        return []


def main():
    # Initialize with nomic-embed-text for better context handling
    # It's slower but handles long documents better
    vector_store = OptimizedVectorStore(
        db_path="./chroma_db_optimized",
        embedding_model="nomic-embed-text"  # Better for long documents
    )
    
    # Process folder (incremental - only new files)
    data_folder = "./cardiovascular_docs"
    
    print("=" * 50)
    print("Starting document processing...")
    
    # Use smaller batch size initially
    retriever = vector_store.process_folder(
        data_folder, 
        force_reprocess=False,  # Change to True to reprocess all files
        batch_size=30,          # Smaller batches to avoid context length errors
        max_workers=1           # Start with 1 worker, increase if needed
    )
    
    if retriever:
        print("\n✓ Vector store ready!")
        
        # Test with a sample query
        print("\nTesting with sample query...")
        results = vector_store.query("What is cardiovascular disease?", retriever)
        print(f"Retrieved {len(results)} documents")
        
        if results:
            print("\nFirst result preview:")
            print(results[0].page_content[:200] + "...")
            
            # Show metadata
            print(f"Source: {results[0].metadata.get('source', 'Unknown')}")
    else:
        print("No retriever available. Check if documents were processed.")


if __name__ == "__main__":
    main()