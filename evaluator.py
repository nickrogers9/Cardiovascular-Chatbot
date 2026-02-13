import json
import time
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class ModelEvaluator:
    def __init__(self, db_path: str = "./chroma_db_optimized"):
        """Initialize the model evaluator with vector store."""
        self.db_path = db_path
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Load evaluation dataset
        with open('evaluation_dataset.json', 'r') as f:
            self.dataset = json.load(f)['questions']
        
        # Load retriever
        self.retriever = self._load_retriever()
        
        # Define models to evaluate
        self.models = ["llama2:latest", "mistral:latest", "olmo2:latest"]
        
        # Results storage
        self.results = {model: [] for model in self.models}
        self.metrics_summary = {}
        
        # Prompt template
        self.template = """
        You are a medical AI assistant specializing in cardiovascular diseases.

        IMPORTANT: You MUST base your answers ONLY on the provided context from medical documents.
        If the context doesn't contain relevant information, say "Based on the provided medical documents, I don't have specific information about that."

        Context from medical documents:
        {context}

        Question: {question}

        Provide a clear, detailed answer based only on the context above. Do not make up information.
        Include relevant statistics or findings when available.

        Answer:"""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
    
    def _load_retriever(self):
        """Load the existing vector store."""
        try:
            vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name="medical_documents"
            )
            return vector_store.as_retriever(search_kwargs={"k": 5})
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def _get_context(self, question: str) -> str:
        """Retrieve context for a question."""
        if self.retriever:
            docs = self.retriever.invoke(question)
            return "\n\n".join([doc.page_content for doc in docs])
        return ""
    
    def _get_model(self, model_name: str):
        """Get Ollama model instance."""
        return OllamaLLM(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for analysis."""
        return word_tokenize(text.lower())
    
    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key medical entities from text."""
        # Simple pattern matching for medical terms
        patterns = [
            r'\b\d+\.?\d*\s*(?:mmHg|mg/dL|mmol/L|m/s|years?|months?|weeks?|days?)\b',
            r'\b(?:ACEi|ARB|ARNi|SGLT2i|GLP-1|MRA|CCB|LDL-C|HDL-C|BNP|NT-proBNP|LVEF|VHD|TAVI|SAVR|ICD)\b',
            r'\b(?:statin|beta[-\s]?blocker|calcium\s+channel\s+blocker|nitrate|anticoagulant|diuretic)\b',
            r'\b(?:hypertension|hyperlipidemia|diabetes|angina|myocardial\s+infarction|heart\s+failure|stroke)\b',
            r'\b\d+\s*%\b'
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _calculate_entity_overlap(self, answer_entities: List[str], truth_entities: List[str]) -> float:
        """Calculate entity overlap between answer and ground truth."""
        if not truth_entities:
            return 0.0
        
        overlap = len(set(answer_entities) & set(truth_entities))
        return overlap / len(truth_entities)
    
    def _check_hallucination(self, answer: str, context: str, ground_truth: str) -> Dict:
        """Check for hallucinations in the answer."""
        # Extract entities from answer, context, and ground truth
        answer_entities = self._extract_key_entities(answer)
        context_entities = self._extract_key_entities(context)
        truth_entities = self._extract_key_entities(ground_truth)
        
        # Find entities in answer that are NOT in context
        hallucinations = [entity for entity in answer_entities 
                        if entity not in context_entities and entity not in truth_entities]
        
        # Check for contradictory statements
        contradictions = 0
        answer_lower = answer.lower()
        
        # Common contradiction patterns
        contradiction_indicators = [
            ("not recommended", "recommended"),
            ("should not", "should"),
            ("contraindicated", "indicated"),
            ("increase", "decrease"),
            ("higher", "lower")
        ]
        
        for neg, pos in contradiction_indicators:
            if neg in answer_lower and pos in answer_lower:
                contradictions += 1
        
        return {
            "hallucination_entities": hallucinations,
            "hallucination_count": len(hallucinations),
            "contradiction_count": contradictions,
            "entity_overlap_with_context": self._calculate_entity_overlap(answer_entities, context_entities),
            "entity_overlap_with_truth": self._calculate_entity_overlap(answer_entities, truth_entities)
        }
    
    def _evaluate_answer(self, question: str, answer: str, ground_truth: str, context: str, 
                        latency: float, token_count: int) -> Dict:
        """Comprehensive evaluation of a single answer."""
        
        # 1. Basic metrics
        throughput = token_count / latency if latency > 0 else 0
        
        # 2. Text similarity metrics
        cosine_sim = self._calculate_cosine_similarity(answer, ground_truth)
        
        # 3. Hallucination analysis
        hallucination_info = self._check_hallucination(answer, context, ground_truth)
        
        # 4. Precision (using entity matching)
        answer_entities = self._extract_key_entities(answer)
        truth_entities = self._extract_key_entities(ground_truth)
        
        if truth_entities:
            correct_entities = len(set(answer_entities) & set(truth_entities))
            total_entities_in_answer = len(answer_entities)
            precision = correct_entities / total_entities_in_answer if total_entities_in_answer > 0 else 0
        else:
            precision = 0
        
        # 5. Answer length and complexity
        answer_words = len(answer.split())
        ground_truth_words = len(ground_truth.split())
        length_ratio = answer_words / ground_truth_words if ground_truth_words > 0 else 0
        
        return {
            "question_id": question.get("id"),
            "question": question.get("question"),
            "answer": answer,
            "ground_truth": ground_truth,
            "context": context[:500] + "..." if len(context) > 500 else context,
            "latency_seconds": latency,
            "token_count": token_count,
            "throughput_tokens_per_sec": throughput,
            "cosine_similarity": cosine_sim,
            "precision": precision,
            "hallucination_entities": hallucination_info["hallucination_entities"],
            "hallucination_count": hallucination_info["hallucination_count"],
            "contradiction_count": hallucination_info["contradiction_count"],
            "entity_overlap_context": hallucination_info["entity_overlap_with_context"],
            "entity_overlap_truth": hallucination_info["entity_overlap_with_truth"],
            "answer_length_words": answer_words,
            "ground_truth_length_words": ground_truth_words,
            "length_ratio": length_ratio,
            "has_hallucination": hallucination_info["hallucination_count"] > 0 or hallucination_info["contradiction_count"] > 0
        }
    
    def evaluate_model(self, model_name: str) -> List[Dict]:
        """Evaluate a single model on all questions."""
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        model_results = []
        model = self._get_model(model_name)
        
        for i, qa in enumerate(self.dataset, 1):
            print(f"\nQuestion {i}/{len(self.dataset)}: {qa['question'][:80]}...")
            
            # Get context
            context = self._get_context(qa['question'])
            
            if not context:
                print(f"  ⚠️ No context found for question {i}")
                continue
            
            # Generate answer with timing
            start_time = time.time()
            try:
                chain = self.prompt | model
                result = chain.invoke({
                    "context": context,
                    "question": qa['question']
                })
                end_time = time.time()
                
                latency = end_time - start_time
                
                # Estimate token count (approximate)
                token_count = len(result.split()) * 1.3  # Rough approximation
                
                # Evaluate the answer
                evaluation = self._evaluate_answer(
                    question=qa,
                    answer=result,
                    ground_truth=qa['ground_truth'],
                    context=context,
                    latency=latency,
                    token_count=token_count
                )
                
                model_results.append(evaluation)
                
                print(f"  ✓ Answer generated in {latency:.2f}s")
                print(f"  Cosine similarity with ground truth: {evaluation['cosine_similarity']:.3f}")
                
            except Exception as e:
                print(f"  ✗ Error generating answer: {e}")
                model_results.append({
                    "question_id": qa["id"],
                    "question": qa["question"],
                    "error": str(e),
                    "latency_seconds": 0,
                    "cosine_similarity": 0
                })
        
        return model_results
    
    def run_evaluation(self):
        """Run evaluation on all models."""
        print("Starting comprehensive model evaluation...")
        print(f"Number of questions: {len(self.dataset)}")
        print(f"Models to evaluate: {self.models}")
        
        all_results = {}
        
        for model in self.models:
            results = self.evaluate_model(model)
            self.results[model] = results
            all_results[model] = results
            
            # Calculate summary statistics
            self._calculate_model_summary(model, results)
        
        # Save results
        self._save_results(all_results)
        
        # Generate reports
        self._generate_summary_report()
        self._generate_detailed_report()
        self._generate_visualizations()
        
        return all_results
    
    def _calculate_model_summary(self, model_name: str, results: List[Dict]):
        """Calculate summary statistics for a model."""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            self.metrics_summary[model_name] = {
                "error": "No valid results"
            }
            return
        
        summary = {
            "model": model_name,
            "total_questions": len(self.dataset),
            "successful_answers": len(valid_results),
            "success_rate": len(valid_results) / len(self.dataset) * 100,
            "avg_latency": np.mean([r.get("latency_seconds", 0) for r in valid_results]),
            "avg_throughput": np.mean([r.get("throughput_tokens_per_sec", 0) for r in valid_results]),
            "avg_cosine_similarity": np.mean([r.get("cosine_similarity", 0) for r in valid_results]),
            "avg_precision": np.mean([r.get("precision", 0) for r in valid_results]),
            "avg_entity_overlap_truth": np.mean([r.get("entity_overlap_truth", 0) for r in valid_results]),
            "hallucination_rate": np.mean([1 if r.get("has_hallucination", False) else 0 for r in valid_results]) * 100,
            "avg_hallucination_count": np.mean([r.get("hallucination_count", 0) for r in valid_results]),
            "avg_answer_length": np.mean([r.get("answer_length_words", 0) for r in valid_results]),
            "median_latency": np.median([r.get("latency_seconds", 0) for r in valid_results]),
            "median_cosine_similarity": np.median([r.get("cosine_similarity", 0) for r in valid_results]),
            "latency_std": np.std([r.get("latency_seconds", 0) for r in valid_results]),
            "cosine_sim_std": np.std([r.get("cosine_similarity", 0) for r in valid_results])
        }
        
        self.metrics_summary[model_name] = summary
    
    def _save_results(self, all_results: Dict):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f'evaluation_results_detailed_{timestamp}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save summary
        with open(f'evaluation_results_summary_{timestamp}.json', 'w') as f:
            json.dump(self.metrics_summary, f, indent=2)
        
        print(f"\nResults saved with timestamp: {timestamp}")
    
    def _generate_summary_report(self):
        """Generate a summary report as text."""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY REPORT")
        print("="*80)
        
        for model, summary in self.metrics_summary.items():
            print(f"\n{'='*60}")
            print(f"MODEL: {model}")
            print(f"{'='*60}")
            
            if "error" in summary:
                print(f"Error: {summary['error']}")
                continue
            
            print(f"Success Rate: {summary['success_rate']:.1f}% ({summary['successful_answers']}/{summary['total_questions']})")
            print(f"Average Latency: {summary['avg_latency']:.2f} seconds")
            print(f"Average Throughput: {summary['avg_throughput']:.1f} tokens/sec")
            print(f"Average Cosine Similarity (to ground truth): {summary['avg_cosine_similarity']:.3f}")
            print(f"Average Precision: {summary['avg_precision']:.3f}")
            print(f"Average Entity Overlap with Truth: {summary['avg_entity_overlap_truth']:.3f}")
            print(f"Hallucination Rate: {summary['hallucination_rate']:.1f}%")
            print(f"Average Hallucinations per Answer: {summary['avg_hallucination_count']:.2f}")
            print(f"Average Answer Length: {summary['avg_answer_length']:.0f} words")
    
    def _generate_detailed_report(self):
        """Generate a detailed CSV report."""
        all_rows = []
        
        for model, results in self.results.items():
            for result in results:
                row = {
                    "model": model,
                    "question_id": result.get("question_id"),
                    "question": result.get("question", "")[:100],
                    "answer_length": result.get("answer_length_words", 0),
                    "latency": result.get("latency_seconds", 0),
                    "throughput": result.get("throughput_tokens_per_sec", 0),
                    "cosine_similarity": result.get("cosine_similarity", 0),
                    "precision": result.get("precision", 0),
                    "entity_overlap_truth": result.get("entity_overlap_truth", 0),
                    "hallucination_count": result.get("hallucination_count", 0),
                    "has_hallucination": result.get("has_hallucination", False),
                    "error": result.get("error", "")
                }
                all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'evaluation_detailed_{timestamp}.csv', index=False)
        
        # Print top 5 questions by cosine similarity for each model
        print("\n" + "="*80)
        print("TOP 5 QUESTIONS BY COSINE SIMILARITY PER MODEL")
        print("="*80)
        
        for model in self.models:
            model_df = df[df['model'] == model]
            if len(model_df) > 0:
                print(f"\n{model}:")
                top_5 = model_df.nlargest(5, 'cosine_similarity')[['question_id', 'cosine_similarity']]
                print(top_5.to_string(index=False))
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary DataFrame
        summary_data = []
        for model, metrics in self.metrics_summary.items():
            if "error" not in metrics:
                summary_data.append(metrics)
        
        if not summary_data:
            print("No valid data for visualizations")
            return
        
        df_summary = pd.DataFrame(summary_data)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Model Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
        
        # 1. Latency comparison
        axes[0, 0].bar(df_summary['model'], df_summary['avg_latency'], color='skyblue')
        axes[0, 0].set_title('Average Latency (seconds)')
        axes[0, 0].set_ylabel('Seconds')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Throughput comparison
        axes[0, 1].bar(df_summary['model'], df_summary['avg_throughput'], color='lightgreen')
        axes[0, 1].set_title('Average Throughput (tokens/sec)')
        axes[0, 1].set_ylabel('Tokens/Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Cosine similarity comparison
        axes[0, 2].bar(df_summary['model'], df_summary['avg_cosine_similarity'], color='salmon')
        axes[0, 2].set_title('Average Cosine Similarity')
        axes[0, 2].set_ylabel('Similarity Score (0-1)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Precision comparison
        axes[1, 0].bar(df_summary['model'], df_summary['avg_precision'], color='gold')
        axes[1, 0].set_title('Average Precision')
        axes[1, 0].set_ylabel('Precision Score (0-1)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)
        
        # 5. Entity overlap comparison
        axes[1, 1].bar(df_summary['model'], df_summary['avg_entity_overlap_truth'], color='violet')
        axes[1, 1].set_title('Average Entity Overlap with Truth')
        axes[1, 1].set_ylabel('Overlap Score (0-1)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)
        
        # 6. Hallucination rate comparison
        axes[1, 2].bar(df_summary['model'], df_summary['hallucination_rate'], color='coral')
        axes[1, 2].set_title('Hallucination Rate (%)')
        axes[1, 2].set_ylabel('Percentage')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. Success rate comparison
        axes[2, 0].bar(df_summary['model'], df_summary['success_rate'], color='lightblue')
        axes[2, 0].set_title('Success Rate (%)')
        axes[2, 0].set_ylabel('Percentage')
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].set_ylim(0, 100)
        
        # 8. Answer length comparison
        axes[2, 1].bar(df_summary['model'], df_summary['avg_answer_length'], color='orange')
        axes[2, 1].set_title('Average Answer Length (words)')
        axes[2, 1].set_ylabel('Word Count')
        axes[2, 1].tick_params(axis='x', rotation=45)
        
        # 9. Combined score (weighted average of key metrics)
        df_summary['combined_score'] = (
            df_summary['avg_cosine_similarity'] * 0.3 +
            df_summary['avg_precision'] * 0.3 +
            (1 - df_summary['hallucination_rate'] / 100) * 0.2 +
            (1 / (df_summary['avg_latency'] + 0.1)) * 0.1 +
            df_summary['success_rate'] / 100 * 0.1
        )
        
        axes[2, 2].bar(df_summary['model'], df_summary['combined_score'], color='green')
        axes[2, 2].set_title('Combined Performance Score')
        axes[2, 2].set_ylabel('Score (0-1, higher is better)')
        axes[2, 2].tick_params(axis='x', rotation=45)
        axes[2, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'model_evaluation_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'model_evaluation_plots_{timestamp}.pdf', bbox_inches='tight')
        plt.show()
        
        # Create a radar chart for comprehensive comparison
        self._create_radar_chart(df_summary, timestamp)
    
    def _create_radar_chart(self, df_summary: pd.DataFrame, timestamp: str):
        """Create a radar chart for model comparison."""
        # Normalize metrics for radar chart
        metrics_to_plot = [
            'avg_cosine_similarity',
            'avg_precision',
            'success_rate',
            'avg_throughput',
            'avg_entity_overlap_truth'
        ]
        
        # Invert latency (lower is better)
        df_summary['inverse_latency'] = 1 / (df_summary['avg_latency'] + 0.1)
        metrics_to_plot.append('inverse_latency')
        
        # Invert hallucination rate (lower is better)
        df_summary['inverse_hallucination'] = 1 - (df_summary['hallucination_rate'] / 100)
        metrics_to_plot.append('inverse_hallucination')
        
        # Normalize each metric to 0-1 scale
        normalized_data = {}
        for metric in metrics_to_plot:
            min_val = df_summary[metric].min()
            max_val = df_summary[metric].max()
            if max_val > min_val:
                normalized_data[metric] = (df_summary[metric] - min_val) / (max_val - min_val)
            else:
                normalized_data[metric] = df_summary[metric] / max_val if max_val > 0 else 0
        
        normalized_df = pd.DataFrame(normalized_data)
        normalized_df['model'] = df_summary['model']
        
        # Create radar chart
        labels = [
            'Similarity', 'Precision', 'Success Rate',
            'Throughput', 'Entity Match', 'Speed (1/latency)', 'Accuracy (1-hallucination)'
        ]
        
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each model
        colors = ['b', 'r', 'g', 'y', 'm', 'c']
        for idx, (_, row) in enumerate(normalized_df.iterrows()):
            values = row[metrics_to_plot].values.tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(f'radar_chart_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'radar_chart_{timestamp}.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualizations saved as:")
        print(f"- model_evaluation_plots_{timestamp}.png/.pdf")
        print(f"- radar_chart_{timestamp}.png/.pdf")

def main():
    """Main function to run the evaluation."""
    print("Cardiovascular Chatbot Model Evaluation System")
    print("="*60)
    
    # Check if vector store exists
    db_path = "./chroma_db_optimized"
    if not os.path.exists(db_path):
        print(f"Error: Vector store not found at {db_path}")
        print("Please run vector.py first to create the vector store.")
        return
    
    # Check if evaluation dataset exists
    if not os.path.exists('evaluation_dataset.json'):
        print("Error: evaluation_dataset.json not found")
        print("Please create the evaluation dataset file first.")
        return
    
    # Initialize and run evaluation
    evaluator = ModelEvaluator(db_path)
    
    try:
        results = evaluator.run_evaluation()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        
        # Print final ranking
        print("\nFINAL MODEL RANKING (by Combined Score):")
        print("-"*60)
        
        rankings = []
        for model, summary in evaluator.metrics_summary.items():
            if "error" not in summary:
                combined_score = (
                    summary['avg_cosine_similarity'] * 0.3 +
                    summary['avg_precision'] * 0.3 +
                    (1 - summary['hallucination_rate'] / 100) * 0.2 +
                    (1 / (summary['avg_latency'] + 0.1)) * 0.1 +
                    summary['success_rate'] / 100 * 0.1
                )
                rankings.append({
                    'model': model,
                    'score': combined_score,
                    'similarity': summary['avg_cosine_similarity'],
                    'hallucination': summary['hallucination_rate'],
                    'latency': summary['avg_latency']
                })
        
        # Sort by combined score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        for i, rank in enumerate(rankings, 1):
            print(f"{i}. {rank['model']}")
            print(f"   Score: {rank['score']:.3f}")
            print(f"   Similarity: {rank['similarity']:.3f}")
            print(f"   Hallucination Rate: {rank['hallucination']:.1f}%")
            print(f"   Latency: {rank['latency']:.2f}s")
            print()
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()