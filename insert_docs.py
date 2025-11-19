"""
insert_docs.py
--------------
Command-line utility to read markdown files from the output directory, chunk them into <1000 character blocks
by header hierarchy, and insert all chunks into ChromaDB with metadata.

Usage:
    python insert_docs.py [--output-dir ...] [--collection ...] [--db-dir ...] [--embedding-model ...]
"""
import os
import argparse
import sys
import re
from typing import List, Dict, Any
from pathlib import Path
from utils import get_chroma_client, get_or_create_collection, add_documents_to_collection
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")


def smart_chunk_markdown(markdown: str, max_len: int = 1000) -> List[str]:
    """Hierarchically splits markdown by #, ##, ### headers, then by characters, to ensure all chunks < max_len."""
    def split_by_header(md, header_pattern):
        indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
        indices.append(len(md))
        return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]

    chunks = []

    for h1 in split_by_header(markdown, r'^# .+$'):
        if len(h1) > max_len:
            for h2 in split_by_header(h1, r'^## .+$'):
                if len(h2) > max_len:
                    for h3 in split_by_header(h2, r'^### .+$'):
                        if len(h3) > max_len:
                            for i in range(0, len(h3), max_len):
                                chunks.append(h3[i:i+max_len].strip())
                        else:
                            chunks.append(h3)
                else:
                    chunks.append(h2)
        else:
            chunks.append(h1)

    final_chunks = []

    for c in chunks:
        if len(c) > max_len:
            final_chunks.extend([c[i:i+max_len].strip() for i in range(0, len(c), max_len)])
        else:
            final_chunks.append(c)

    return [c for c in final_chunks if c]

def load_markdown_files(output_dir: str) -> List[Dict[str, str]]:
    """Load markdown files from output directory, extracting frontmatter and content.
    
    Args:
        output_dir: Directory containing markdown files
        
    Returns:
        List of dicts with 'url' and 'markdown' keys
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory '{output_dir}' does not exist.")
        return []
    
    results = []
    
    # Find all .md files in the output directory
    md_files = list(output_path.glob("*.md"))
    
    if not md_files:
        print(f"No markdown files found in '{output_dir}'.")
        return []
    
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse YAML frontmatter
            # Look for frontmatter between --- markers
            frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
            match = re.match(frontmatter_pattern, content, re.DOTALL)
            
            if match:
                frontmatter_text = match.group(1)
                markdown_content = match.group(2)
                
                # Extract source_url from frontmatter
                url_match = re.search(r'source_url:\s*(.+)', frontmatter_text)
                if url_match:
                    source_url = url_match.group(1).strip()
                else:
                    # Fallback: use filename if source_url not found
                    source_url = str(filepath)
            else:
                # No frontmatter found, use entire content as markdown
                markdown_content = content
                source_url = str(filepath)
            
            if markdown_content.strip():
                results.append({
                    'url': source_url,
                    'markdown': markdown_content
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    return results

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def main():
    parser = argparse.ArgumentParser(description="Insert markdown files from output directory into ChromaDB")
    parser.add_argument("--output-dir", default="./output", help="Directory containing markdown files")
    parser.add_argument("--collection", default="docs", help="ChromaDB collection name")
    parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--batch-size", type=int, default=100, help="ChromaDB insert batch size")
    args = parser.parse_args()

    # Load markdown files from output directory
    print(f"Loading markdown files from '{args.output_dir}'...")
    crawl_results = load_markdown_files(args.output_dir)
    
    if not crawl_results:
        print("No markdown files found to process.")
        sys.exit(1)
    
    print(f"Found {len(crawl_results)} markdown files.")

    # Chunk and collect metadata
    ids, documents, metadatas = [], [], []
    chunk_idx = 0
    for doc in crawl_results:
        url = doc['url']
        md = doc['markdown']
        chunks = smart_chunk_markdown(md, max_len=args.chunk_size)
        for chunk in chunks:
            ids.append(f"chunk-{chunk_idx}")
            documents.append(chunk)
            meta = extract_section_info(chunk)
            meta["chunk_index"] = chunk_idx
            meta["source"] = url
            metadatas.append(meta)
            chunk_idx += 1

    if not documents:
        print("No documents found to insert.")
        sys.exit(1)

    print(f"Inserting {len(documents)} chunks into ChromaDB collection '{args.collection}'...")

    client = get_chroma_client(args.db_dir)
    collection = get_or_create_collection(client, args.collection, embedding_model_name=args.embedding_model)
    add_documents_to_collection(collection, ids, documents, metadatas, batch_size=args.batch_size)

    print(f"Successfully added {len(documents)} chunks to ChromaDB collection '{args.collection}'.")

if __name__ == "__main__":
    main()
