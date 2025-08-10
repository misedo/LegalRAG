#!/usr/bin/env python3

import requests
import json
import time
import os
import argparse
import hashlib
from datetime import datetime
import pandas as pd
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import pickle
from tqdm import tqdm
import numpy as np

# Optional: For OpenAI embeddings
try:
    from openai import OpenAI
    client=OpenAI()
    
    #client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Using local embeddings.")
    from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION
# ============================================

# Your CourtListener API Token
COURTLISTENER_TOKEN = "2..........8"

# OpenAI API Key (for embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# API Headers with authentication
HEADERS = {
    'Authorization': f'Token {COURTLISTENER_TOKEN}'
}

# Data directories
DATA_DIR = "./legal_rag_data"
DB_DIR = f"{DATA_DIR}/chromadb"
CASES_DIR = f"{DATA_DIR}/cases"
METADATA_FILE = f"{DATA_DIR}/metadata.json"

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(CASES_DIR, exist_ok=True)

# ============================================
# COURTLISTENER DATA DOWNLOAD
# ============================================

class CourtListenerDownloader:
    def __init__(self, token: str):
        self.token = token
        self.headers = {'Authorization': f'Token {token}'}
        self.base_url = "https://www.courtlistener.com/api/rest/v4/"

    def download_cases(self, max_cases: int = 500, verbose: bool = True) -> List[Dict]:
        """Download cases from multiple sources for variety"""
        all_cases = []

        # 1. Get clusters (case groups)
        if verbose:
            print("\n📥 Fetching case clusters from CourtListener...")
        clusters = self._fetch_clusters(min(max_cases // 2, 250))
        all_cases.extend(clusters)

        # 2. Get opinions
        if len(all_cases) < max_cases and verbose:
            print("\n📥 Fetching additional opinions...")
        opinions = self._fetch_opinions(min(max_cases - len(all_cases), 250))
        all_cases.extend(opinions)

        # 3. Search for specific case types
        if len(all_cases) < max_cases and verbose:
            print("\n📥 Searching for specific case types...")
        search_cases = self._search_specific_cases(max_cases - len(all_cases))
        all_cases.extend(search_cases)

        # Remove duplicates
        unique_cases = self._remove_duplicates(all_cases)

        if verbose:
            print(f"\n✓ Total unique cases collected: {len(unique_cases)}")

        return unique_cases[:max_cases]

    def _fetch_clusters(self, limit: int) -> List[Dict]:
        """Fetch case clusters"""
        cases = []
        url = f"{self.base_url}clusters/"

        params = {
            'page_size': min(100, limit),
            'order_by': '-date_filed'
        }

        try:
            while url and len(cases) < limit:
                response = requests.get(url, headers=self.headers, params=params, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])

                    for cluster in results:
                        if len(cases) >= limit:
                            break
                        case = self._process_cluster(cluster)
                        if case:
                            cases.append(case)

                    url = data.get('next')
                    params = {}  # Clear params for pagination
                    time.sleep(0.5)  # Rate limiting
                else:
                    print(f"  Error fetching clusters: {response.status_code}")
                    break

        except Exception as e:
            print(f"  Exception fetching clusters: {str(e)[:100]}")

        print(f"  Fetched {len(cases)} clusters")
        return cases

    def _fetch_opinions(self, limit: int) -> List[Dict]:
        """Fetch opinions"""
        cases = []
        url = f"{self.base_url}opinions/"

        params = {
            'page_size': min(100, limit),
            'order_by': '-date_created'
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=20)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                for opinion in results[:limit]:
                    case = self._process_opinion(opinion)
                    if case:
                        cases.append(case)

        except Exception as e:
            print(f"  Exception fetching opinions: {str(e)[:100]}")

        print(f"  Fetched {len(cases)} opinions")
        return cases

    def _search_specific_cases(self, limit: int) -> List[Dict]:
        """Search for specific types of cases"""
        cases = []
        search_queries = [
            "breach of contract",
            "negligence",
            "intellectual property",
            "employment discrimination",
            "fiduciary duty",
            "medical malpractice",
            "patent software",
            "securities fraud",
            "antitrust",
            "constitutional rights"
        ]

        per_query_limit = max(10, limit // len(search_queries))

        for query in search_queries:
            if len(cases) >= limit:
                break

            try:
                url = "https://www.courtlistener.com/api/rest/v3/search/"
                params = {
                    'q': query,
                    'type': 'o',
                    'order_by': 'score desc',
                    'stat_Precedential': 'on'
                }

                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])[:per_query_limit]

                    for result in results:
                        case = self._process_search_result(result)
                        if case:
                            cases.append(case)

                    print(f"  Found {len(results)} cases for '{query}'")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"  Search error for '{query}': {str(e)[:50]}")

        return cases

    def _process_cluster(self, cluster: Dict) -> Optional[Dict]:
        """Process a cluster into case format"""
        try:
            citations = []
            for field in ['federal_cite_one', 'federal_cite_two', 'federal_cite_three',
                         'state_cite_one', 'state_cite_two', 'neutral_cite']:
                if cluster.get(field):
                    citations.append(str(cluster[field]))

            text_parts = []
            for field in ['syllabus', 'summary', 'procedural_history']:
                if cluster.get(field):
                    text_parts.append(cluster[field])

            return {
                'id': f"cluster_{cluster.get('id', '')}",
                'case_name': cluster.get('case_name', '') or cluster.get('case_name_full', ''),
                'court': cluster.get('court', ''),
                'date_filed': cluster.get('date_filed', ''),
                'citations': ' | '.join(citations),
                'precedential_status': cluster.get('precedential_status', ''),
                'text': ' '.join(text_parts)[:3000],
                'case_type': 'cluster',
                'url': cluster.get('absolute_url', '')
            }
        except Exception as e:
            return None

    def _process_opinion(self, opinion: Dict) -> Optional[Dict]:
        """Process an opinion into case format"""
        try:
            text = opinion.get('plain_text', '') or opinion.get('html', '') or opinion.get('html_lawbox', '')
            text = self._clean_text(text)

            return {
                'id': f"opinion_{opinion.get('id', '')}",
                'case_name': self._extract_case_name(opinion),
                'court': opinion.get('court', ''),
                'date_filed': opinion.get('date_created', ''),
                'citations': '',  # Opinions might not have citations
                'text': text[:3000],
                'case_type': 'opinion',
                'author': opinion.get('author_str', ''),
                'url': opinion.get('absolute_url', '')
            }
        except Exception as e:
            return None

    def _process_search_result(self, result: Dict) -> Optional[Dict]:
        """Process a search result into case format"""
        try:
            citations = result.get('citation', [])
            if isinstance(citations, list):
                citations = ' | '.join([str(c) for c in citations])
            else:
                citations = str(citations) if citations else ''

            return {
                'id': f"search_{result.get('id', '')}",
                'case_name': result.get('caseName', ''),
                'court': result.get('court', ''),
                'date_filed': result.get('dateFiled', ''),
                'citations': citations,
                'text': self._clean_text(result.get('text', result.get('snippet', '')))[:3000],
                'case_type': 'search',
                'url': result.get('absolute_url', '')
            }
        except Exception as e:
            return None

    def _extract_case_name(self, opinion: Dict) -> str:
        """Extract case name from opinion"""
        for field in ['case_name', 'case_name_full', 'case_name_short']:
            if opinion.get(field):
                return opinion[field]
        return f"Opinion {opinion.get('id', 'Unknown')}"

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
        # Remove HTML tags
        import re
        text = re.sub('<[^<]+?>', '', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text

    def _remove_duplicates(self, cases: List[Dict]) -> List[Dict]:
        """Remove duplicate cases"""
        seen = set()
        unique = []
        for case in cases:
            case_key = f"{case.get('case_name', '')}{case.get('date_filed', '')}"
            if case_key not in seen and case.get('case_name'):
                seen.add(case_key)
                unique.append(case)
        return unique

# ============================================
# VECTOR DATABASE BUILDER
# ============================================

class VectorDatabaseBuilder:
    def __init__(self, persist_directory: str = DB_DIR, use_openai: bool = True):
        self.persist_directory = persist_directory
        print("\n🤖 Initializing Vector Database Builder...")

        # Choose embedding method
        self.use_openai = use_openai and OPENAI_AVAILABLE and OPENAI_API_KEY

        if self.use_openai:
            # Use OpenAI embeddings
            print("  Using OpenAI embeddings (text-embedding-3-small)")
            print(f"  API Key configured: {OPENAI_API_KEY[:10]}...")
            self.embedding_model = "text-embedding-3-small"
            self.embedding_dimension = 1536  # OpenAI ada-002 dimension
        else:
            # Fallback to local embeddings
            print("  Loading local embedding model (all-MiniLM-L6-v2)...")
            print("  To use OpenAI embeddings, set OPENAI_API_KEY environment variable")
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384  # MiniLM dimension

        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI or local model."""
        if self.use_openai:
            from openai import RateLimitError, APIError

            embeddings = []
            batch_size = 50  # Adjust for your rate limit comfort

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                attempt = 0

                while True:
                    try:
                        response = client.embeddings.create(
                            input=batch,
                            model=self.embedding_model
                        )
                        # ✅ Correct way to get embeddings from new OpenAI SDK
                        batch_embeddings = [item.embedding for item in response.data]
                        embeddings.extend(batch_embeddings)

                        if len(texts) > batch_size:
                            print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts...")

                        break  # Exit retry loop

                    except RateLimitError:
                        attempt += 1
                        wait_time = min(5 * attempt, 60)
                        print(f"    ⏳ Rate limit hit. Retrying in {wait_time}s...")
                        time.sleep(wait_time)

                    except APIError as e:
                        print(f"    ❌ OpenAI API error: {e}. Retrying in 10s...")
                        time.sleep(10)

                    except Exception as e:
                        print(f"    ❌ Unexpected embedding error: {e}")
                        raise

            return embeddings

        else:
            # Local embeddings (SentenceTransformers)
            return self.embedder.encode(texts, show_progress_bar=True).tolist()


    def build_database(self, cases: List[Dict], rebuild: bool = False) -> None:
        """Build or update the vector database"""
        collection_name = "legal_cases"

        # Handle existing collection
        try:
            collection = self.chroma_client.get_collection(collection_name)
            if rebuild:
                print(f"  Deleting existing collection with {collection.count()} documents...")
                self.chroma_client.delete_collection(collection_name)
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                print(f"  Using existing collection with {collection.count()} documents")
        except:
            print("  Creating new collection...")
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # Prepare documents for indexing
        print("\n📊 Processing cases for vector database...")
        documents = []
        metadatas = []
        ids = []

        for case in tqdm(cases, desc="Preparing cases"):
            # Create searchable text
            search_text = self._create_search_text(case)
            documents.append(search_text)

            # Prepare metadata (ensure all values are strings)
            metadatas.append({
                'case_name': str(case.get('case_name', ''))[:500],
                'court': str(case.get('court', ''))[:200],
                'date_filed': str(case.get('date_filed', ''))[:50],
                'citations': str(case.get('citations', ''))[:500],
                'case_type': str(case.get('case_type', ''))[:50],
                'url': str(case.get('url', ''))[:500]
            })

            ids.append(case['id'])

        # Create embeddings and add to database in batches
        print("\n🔄 Creating embeddings and indexing...")
        print(f"  Using {'OpenAI API' if self.use_openai else 'local model'} for embeddings")

        batch_size = 50 if self.use_openai else 100  # Smaller batches for API

        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing batches"):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            # Create embeddings
            embeddings = self.create_embeddings(batch_docs)

            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

        print(f"\n✓ Successfully indexed {len(documents)} cases")
        print(f"  Total documents in collection: {collection.count()}")
        print(f"  Embedding dimension: {self.embedding_dimension}")
        if self.use_openai:
            print(f"  Estimated cost: ${len(documents) * 0.0001:.2f} (OpenAI embeddings)")

    def _create_search_text(self, case: Dict) -> str:
        """Create comprehensive searchable text"""
        parts = [
            case.get('case_name', ''),
            case.get('court', ''),
            case.get('date_filed', ''),
            case.get('text', ''),
            case.get('citations', '')
        ]
        return ' '.join([str(p) for p in parts if p])[:4000]

    def save_cases_to_disk(self, cases: List[Dict]) -> None:
        """Save cases to disk for later use"""
        # Save as JSON
        with open(f"{CASES_DIR}/cases.json", 'w') as f:
            json.dump(cases, f, indent=2)

        # Save as CSV for easy viewing
        df = pd.DataFrame(cases)
        df.to_csv(f"{CASES_DIR}/cases.csv", index=False)

        # Save metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_cases': len(cases),
            'case_types': list(set(c.get('case_type', '') for c in cases)),
            'courts': list(set(c.get('court', '') for c in cases if c.get('court')))[:20]
        }

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Saved {len(cases)} cases to {CASES_DIR}/")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Download legal cases and build vector database')
    parser.add_argument('--cases', type=int, default=500, help='Number of cases to download')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild database from scratch')
    parser.add_argument('--skip-download', action='store_true', help='Skip download, use existing cases')
    parser.add_argument('--use-local-embeddings', action='store_true', 
                       help='Use local embeddings instead of OpenAI')

    args = parser.parse_args()

    print("="*60)
    print("LEGAL CASE DATABASE BUILDER")
    print("="*60)
    print(f"Configuration:")
    print(f"  Cases to download: {args.cases}")
    print(f"  Rebuild database: {args.rebuild}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Embeddings: {'Local (all-MiniLM-L6-v2)' if args.use_local_embeddings else 'OpenAI (text-embedding-ada-002)'}")
    print("="*60)

    # Download cases (unless skipped)
    if not args.skip_download:
        downloader = CourtListenerDownloader(COURTLISTENER_TOKEN)
        cases = downloader.download_cases(max_cases=args.cases)

        if not cases:
            print("\n❌ Failed to download cases. Check your internet connection and API token.")
            return

        # Save cases to disk
        builder = VectorDatabaseBuilder(use_openai=not args.use_local_embeddings)
        builder.save_cases_to_disk(cases)
    else:
        # Load existing cases
        print("\n📂 Loading existing cases from disk...")
        try:
            with open(f"{CASES_DIR}/cases.json", 'r') as f:
                cases = json.load(f)
            print(f"  Loaded {len(cases)} cases")
        except FileNotFoundError:
            print("❌ No existing cases found. Run without --skip-download first.")
            return

    # Build vector database
    builder = VectorDatabaseBuilder(use_openai=not args.use_local_embeddings)
    builder.build_database(cases, rebuild=args.rebuild)

    # Print summary
    print("\n" + "="*60)
    print("✅ DATABASE BUILD COMPLETE!")
    print("="*60)
    print(f"\nDatabase Statistics:")
    print(f"  Total cases indexed: {len(cases)}")
    print(f"  Database location: {DB_DIR}")
    print(f"  Cases saved to: {CASES_DIR}")

    # Load and print metadata
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"\nMetadata:")
        print(f"  Last updated: {metadata['last_updated']}")
        print(f"  Case types: {', '.join(metadata['case_types'])}")
        print(f"  Number of courts: {len(metadata['courts'])}")
    except:
        pass

    print("\n🚀 You can now run legal_rag_system.py to use the RAG system!")

if __name__ == "__main__":
    main()