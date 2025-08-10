#!/usr/bin/env python3

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from tabulate import tabulate
import textwrap
from dotenv import load_dotenv
load_dotenv()

# Optional imports
import sys

# Debug: check sys.path to ensure we're using the venv site-packages
print(f"[DEBUG] sys.path: {sys.path}")

try:
    from openai import OpenAI
    from openai import RateLimitError
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Failed to import OpenAI: {e}")
    OPENAI_AVAILABLE = False
except Exception as e:
    print(f"[DEBUG] Unexpected error importing OpenAI: {e}")
    OPENAI_AVAILABLE = False


try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ============================================
# CONFIGURATION
# ============================================

DATA_DIR = "./legal_rag_data"
DB_DIR = f"{DATA_DIR}/chromadb"
CASES_DIR = f"{DATA_DIR}/cases"
METADATA_FILE = f"{DATA_DIR}/metadata.json"
RESULTS_DIR = f"{DATA_DIR}/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# LEGAL RAG SYSTEM
# ============================================

class LegalRAGSystem:
    def __init__(self, db_dir: str = DB_DIR, use_llm: bool = False, use_local_embeddings: bool = False):
        print("\n🚀 Initializing Legal RAG System...")

        # Save key for later and debug
        self.openai_key = os.getenv("OPENAI_API_KEY")
        print(f"[DEBUG] OpenAI key detected: {'Yes' if self.openai_key else 'No'}")
        if self.openai_key:
            print(f"[DEBUG] OpenAI key preview: {self.openai_key[:5]}***")
        print(f"[DEBUG] OpenAI package available: {'Yes' if OPENAI_AVAILABLE else 'No'}")
        print(f"[DEBUG] use_local_embeddings flag: {use_local_embeddings}")

        self.use_openai_embeddings = not use_local_embeddings and bool(self.openai_key) and OPENAI_AVAILABLE
        print(f"[DEBUG] use_openai_embeddings decision: {self.use_openai_embeddings}")

        if not os.path.exists(db_dir):
            raise FileNotFoundError(f"Database not found at {db_dir}. Run download_and_index.py first.")

        if self.use_openai_embeddings:
            print("  Using OpenAI embeddings (text-embedding-3-small)")
            self.client = OpenAI(api_key=self.openai_key)
            self.embedding_model = "text-embedding-3-small"
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("SentenceTransformers is not installed. Install with: pip install sentence-transformers")
            print("  Using local embedding model (all-MiniLM-L6-v2)")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print("  Connecting to vector database...")
        self.chroma_client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection("legal_cases")
        print(f"  ✓ Connected to database with {self.collection.count()} cases")

        self.cases_metadata = self._load_cases_metadata()
        self.use_llm = use_llm and bool(self.openai_key) and OPENAI_AVAILABLE
        if self.use_llm:
            print("  ✓ LLM generation enabled (OpenAI)")

        print("\n✅ RAG System Ready!\n")

    def _load_cases_metadata(self) -> Dict:
        metadata = {}
        try:
            with open(f"{CASES_DIR}/cases.json", 'r') as f:
                cases = json.load(f)
                metadata['cases'] = {case['id']: case for case in cases}
            with open(METADATA_FILE, 'r') as f:
                metadata['system'] = json.load(f)
        except FileNotFoundError:
            print("  ⚠️ Metadata files not found. Some features may be limited.")
            metadata = {'cases': {}, 'system': {}}
        return metadata

    def _embed_query(self, query: str) -> List[float]:
        if self.use_openai_embeddings:
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            return resp.data[0].embedding
        else:
            return self.embedder.encode([query])[0].tolist()

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self._embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        retrieved_cases = []
        for i in range(len(results['ids'][0])):
            case_id = results['ids'][0][i]
            full_case = self.cases_metadata.get('cases', {}).get(case_id, {})
            case_info = {
                'id': case_id,
                'case_name': results['metadatas'][0][i].get('case_name', ''),
                'court': results['metadatas'][0][i].get('court', ''),
                'date_filed': results['metadatas'][0][i].get('date_filed', ''),
                'citations': results['metadatas'][0][i].get('citations', ''),
                'similarity_score': float(1 - results['distances'][0][i]),
                'snippet': results['documents'][0][i][:500] if results['documents'][0][i] else '',
                'full_text': full_case.get('text', '')[:2000] if full_case else '',
                'url': results['metadatas'][0][i].get('url', ''),
                'rank': i + 1
            }
            retrieved_cases.append(case_info)
        return retrieved_cases

    def augment(self, query: str, retrieved_cases: List[Dict]) -> str:
        """Improved prompt for LLM with structured case context."""
        case_context = ""
        for case in retrieved_cases:
            case_context += f"""
Case {case['rank']}:
- Name: {case['case_name']}
- Court: {case['court']}
- Date Filed: {case['date_filed']}
- Citations: {case['citations']}
- Relevance Score: {case['similarity_score']:.2%}
- Summary: {case['snippet']}
"""

        augmented_prompt = f"""
You are an expert legal research assistant. Your task is to analyze the following legal cases 
and answer the user's query with precision and legal clarity.

User Query:
"{query}"

Relevant Cases:
{case_context}

Guidelines for your answer:
1. Identify the key legal principles from the relevant cases.
2. Compare and contrast the cases, noting similarities and differences.
3. Cite cases explicitly when making points (use 'Case X' format).
4. If applicable, explain jurisdictional differences.
5. Avoid making up facts or laws not present in the cases.
6. Write in a clear, professional tone suitable for a legal report.
7. Provide actionable insights or recommended next steps.

Now provide your analysis:
"""
        return augmented_prompt

    def generate(self, augmented_prompt: str) -> str:
        if not self.use_llm:
            return self._generate_template_response(augmented_prompt)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert legal research assistant."},
                    {"role": "user", "content": augmented_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            return self._generate_template_response(augmented_prompt)

    def _generate_template_response(self, augmented_prompt: str) -> str:
        lines = augmented_prompt.split('\n')
        cases = [line.split(':', 1)[1].strip() for line in lines if line.startswith('Case ')]
        return f"""Based on the retrieved cases, here's a legal research summary:

RELEVANT PRECEDENTS IDENTIFIED:
{chr(10).join(f'• {case}' for case in cases[:5])}

KEY FINDINGS:
These cases establish relevant precedents for your query.

RECOMMENDED ANALYSIS APPROACH:
1. Review the full text of the top-ranked cases
2. Examine legal principles applied
3. Note distinguishing factors
4. Consider jurisdictional variations

NEXT STEPS:
- Analyze cases in detail
- Check for more recent citations
- Verify current legal status
"""

    def rag_query(self, query: str, k: int = 5, verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n🔍 Processing Query: '{query}'")
        retrieved_cases = self.retrieve(query, k=k)
        augmented_prompt = self.augment(query, retrieved_cases)

        # ✅ Show the generated prompt for your assignment
        print("\n📜 GENERATED PROMPT:")
        print("="*60)
        print(augmented_prompt)
        print("="*60)

        generated_answer = self.generate(augmented_prompt)
        return {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'retrieved_cases': retrieved_cases,
            'augmented_prompt': augmented_prompt,
            'generated_answer': generated_answer,
            'metadata': {
                'num_cases_retrieved': k,
                'llm_used': self.use_llm,
                'top_similarity': retrieved_cases[0]['similarity_score'] if retrieved_cases else 0
            }
        }

# ============================================
# INTERACTIVE MODE
# ============================================

def interactive_mode(rag_system: LegalRAGSystem):
    print("\n" + "="*60)
    print("🎯 INTERACTIVE LEGAL RAG SYSTEM")
    print("="*60)
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the system")
    print("  'help' - Show this help message\n")

    while True:
        try:
            query = input("\n⚖️ Legal Query > ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            if query.lower() == 'help':
                print("\nEnter a legal question, e.g.: 'breach of contract software development'")
                continue

            if query:
                result = rag_system.rag_query(query, k=5, verbose=True)
                print("\n📋 GENERATED ANSWER\n" + "="*60)
                print(result['generated_answer'])

                # Show retrieved cases
                print("\n📚 RETRIEVED CASES\n" + "="*60)
                table_data = [
                    [case['rank'], textwrap.fill(case['case_name'][:40], width=40),
                     case['court'][:20], case['date_filed'][:10],
                     f"{case['similarity_score']:.1%}"]
                    for case in result['retrieved_cases']
                ]
                print(tabulate(table_data, headers=['#', 'Case Name', 'Court', 'Date', 'Match'], tablefmt='grid'))

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Legal RAG System')
    parser.add_argument('--query', type=str, help='Single query to process')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for generation')
    parser.add_argument('--use-local-embeddings', action='store_true', help='Use local embeddings instead of OpenAI')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("⚖️  LEGAL RAG SYSTEM")
    print("="*60)

    try:
        rag_system = LegalRAGSystem(
            use_llm=args.use_llm,
            use_local_embeddings=args.use_local_embeddings
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return

    if args.interactive:
        interactive_mode(rag_system)
    elif args.query:
        result = rag_system.rag_query(args.query, k=5, verbose=True)
        print("\n📋 GENERATED ANSWER\n" + "="*60)
        print(result['generated_answer'])

if __name__ == "__main__":
    main()
