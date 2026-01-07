"""
BM25 Keyword Search for hybrid retrieval.

Provides sparse (lexical) search to complement dense (semantic) search.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import math
from collections import Counter


@dataclass
class BM25Result:
    """A single BM25 search result."""
    text: str
    score: float
    doc_hash: str
    chunk_id: int
    filename: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "doc_hash": self.doc_hash,
            "chunk_id": self.chunk_id,
            "filename": self.filename
        }


class BM25Searcher:
    """
    BM25 (Best Matching 25) keyword search implementation.
    
    Features:
    - Efficient inverted index
    - Configurable BM25 parameters
    - Per-document filtering
    - Tokenization with stopword removal
    """
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we',
        'they', 'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'same', 'so', 'than', 'too', 'very'
    }
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 searcher.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0-1)
            epsilon: Floor for IDF calculation
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Index structures
        self._documents: List[Dict[str, Any]] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0
        self._inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, term_freq)]
        self._doc_freqs: Dict[str, int] = {}  # term -> num docs containing term
        self._is_indexed: bool = False
    
    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from document chunks.
        
        Args:
            chunks: List of chunk dicts with 'text', 'doc_hash', 'chunk_id', 'filename'
        """
        self._documents = chunks
        self._doc_lengths = []
        self._inverted_index = {}
        self._doc_freqs = {}
        
        # Build index
        for doc_idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            tokens = self._tokenize(text)
            self._doc_lengths.append(len(tokens))
            
            # Count term frequencies
            term_counts = Counter(tokens)
            
            for term, freq in term_counts.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = []
                    self._doc_freqs[term] = 0
                
                self._inverted_index[term].append((doc_idx, freq))
                self._doc_freqs[term] += 1
        
        # Calculate average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        
        self._is_indexed = True
    
    def search(
        self,
        query: str,
        k: int = 10,
        doc_hash: Optional[str] = None
    ) -> List[BM25Result]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query string
            k: Number of results to return
            doc_hash: Optional filter to specific document
            
        Returns:
            List of BM25Result sorted by score
        """
        if not self._is_indexed:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Calculate scores for all documents
        scores: Dict[int, float] = {}
        n_docs = len(self._documents)
        
        for term in query_tokens:
            if term not in self._inverted_index:
                continue
            
            # Calculate IDF
            doc_freq = self._doc_freqs[term]
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            idf = max(idf, self.epsilon)
            
            # Score documents containing this term
            for doc_idx, term_freq in self._inverted_index[term]:
                # Apply document filter if specified
                if doc_hash and self._documents[doc_idx].get("doc_hash") != doc_hash:
                    continue
                
                # BM25 formula
                doc_len = self._doc_lengths[doc_idx]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                
                score = idf * (numerator / denominator)
                scores[doc_idx] = scores.get(doc_idx, 0) + score
        
        # Sort and return top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_idx, score in sorted_docs:
            chunk = self._documents[doc_idx]
            results.append(BM25Result(
                text=chunk.get("text", ""),
                score=score,
                doc_hash=chunk.get("doc_hash", ""),
                chunk_id=chunk.get("chunk_id", 0),
                filename=chunk.get("filename", "unknown")
            ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing/search.
        
        Args:
            text: Input text
            
        Returns:
            List of lowercase tokens with stopwords removed
        """
        # Lowercase and extract words
        text_lower = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text_lower)
        
        # Remove stopwords and very short tokens
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]
    
    def clear(self) -> None:
        """Clear the index."""
        self._documents = []
        self._doc_lengths = []
        self._avg_doc_length = 0
        self._inverted_index = {}
        self._doc_freqs = {}
        self._is_indexed = False
    
    @property
    def is_indexed(self) -> bool:
        """Check if index is built."""
        return self._is_indexed
    
    @property
    def num_documents(self) -> int:
        """Get number of indexed documents."""
        return len(self._documents)


def reciprocal_rank_fusion(
    vector_results: List[Any],
    keyword_results: List[BM25Result],
    k: int = 60,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Combine vector and keyword search results using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank)) for each result list
    
    Args:
        vector_results: Results from vector search (SearchResult objects)
        keyword_results: Results from BM25 search
        k: RRF constant (default 60)
        vector_weight: Weight for vector search results
        keyword_weight: Weight for keyword search results
        
    Returns:
        Combined and re-ranked results
    """
    # Build score map: chunk_id -> (text, combined_score, metadata)
    scores: Dict[str, Dict[str, Any]] = {}
    
    # Process vector results
    for rank, result in enumerate(vector_results, 1):
        chunk_key = f"{result.doc_hash}:{result.chunk_id}"
        rrf_score = vector_weight / (k + rank)
        
        if chunk_key not in scores:
            scores[chunk_key] = {
                "text": result.text,
                "score": 0,
                "doc_hash": result.doc_hash,
                "chunk_id": result.chunk_id,
                "filename": result.filename,
                "vector_score": result.score,
                "keyword_score": 0,
                "sources": ["vector"]
            }
        
        scores[chunk_key]["score"] += rrf_score
        scores[chunk_key]["vector_score"] = result.score
    
    # Process keyword results
    for rank, result in enumerate(keyword_results, 1):
        chunk_key = f"{result.doc_hash}:{result.chunk_id}"
        rrf_score = keyword_weight / (k + rank)
        
        if chunk_key not in scores:
            scores[chunk_key] = {
                "text": result.text,
                "score": 0,
                "doc_hash": result.doc_hash,
                "chunk_id": result.chunk_id,
                "filename": result.filename,
                "vector_score": 0,
                "keyword_score": 0,
                "sources": []
            }
        
        scores[chunk_key]["score"] += rrf_score
        scores[chunk_key]["keyword_score"] = result.score
        if "keyword" not in scores[chunk_key]["sources"]:
            scores[chunk_key]["sources"].append("keyword")
    
    # Sort by combined score
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    
    return sorted_results
