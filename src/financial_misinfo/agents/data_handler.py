#######################################################################################################################################        
#                                                       DATAHANDLER AGENT                                                             #        
#######################################################################################################################################                
        

import os
import warnings
import logging
import json
import uuid
import time
import asyncio
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import requests
from dataclasses import dataclass, field
from tqdm import tqdm
from langchain_community.llms import HuggingFaceEndpoint
import math
import traceback
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
import csv
import pickle
from typing import List, Dict, Any
from pathlib import Path


class DataHandlerAgent:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_store = None
        self.chunks = []
        self.metadata = []     # Store claim metadata
        
        # Flag to track if we've tried auto-loading
        self._tried_autoload = False
       

    def prepare_documents(self, data: List[Dict]) -> List[str]:
        """
        Prepare documents for RAG by creating separate entries for each evidence sentence
        while maintaining their relationships with sources and metadata.
        """
        documents = []
        self.metadata = []

        for entry_idx, entry in enumerate(data):
            # Extract base metadata that will be common for all evidence sentences
            base_metadata = {
                'claim': entry.get('claim', ''),
                'label': entry.get('label', 'unverified'),
                'justification': entry.get('justification', ''),
                'issues': entry.get('issues', []),
                'posted': entry.get('posted', ''),
                'sci_digest': entry.get('sci_digest', [])
            }

            # Process each evidence entry separately
            if isinstance(entry.get('evidence'), list):
                for ev_idx, ev in enumerate(entry['evidence']):
                    if isinstance(ev, dict) and 'sentence' in ev:
                        # Create document text focused on the specific evidence
                        document_text = (
                            f"CLAIM: {base_metadata['claim']}\n"
                            f"EVIDENCE: {ev['sentence']}\n"
                            f"VERIFICATION: {base_metadata['label']}\n"
                            f"CONTEXT: {base_metadata['justification']}"
                        )

                        # Clean and normalize text
                        cleaned_text = ' '.join(
                            document_text.replace('\n', ' ')
                            .replace('\\n', ' ')
                            .replace('\\t', ' ')
                            .replace('\\r', ' ')
                            .strip()
                            .split()
                        )

                        # Store metadata for this specific evidence
                        evidence_metadata = {
                            'doc_id': f'claim_{entry_idx}_evidence_{ev_idx}',
                            'claim_text': base_metadata['claim'],
                            'evidence_sentence': ev['sentence'],
                            'label': base_metadata['label'],
                            'hrefs': ev.get('hrefs', []),
                            'metadata': base_metadata
                        }

                        documents.append(cleaned_text)
                        self.metadata.append(evidence_metadata)

        return documents

    
    
    def build_index(self, documents: List[str]):
        """
        Build an optimized FAISS index for the prepared documents.
        """
        try:
            if not documents:
                raise ValueError("No documents to index")

            print("Generating embeddings...")
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=32
            )

            dimension = embeddings.shape[1]

            # Create IVF index for efficient similarity search
            quantizer = faiss.IndexFlatL2(dimension)
            self.vector_store = faiss.IndexIVFFlat(
                quantizer, dimension, 
                min(int(np.sqrt(len(documents))), 16384),
                faiss.METRIC_L2
            )
            
            self.vector_store.train(embeddings)
            self.vector_store.add(np.array(embeddings))
            self.chunks = documents

            print(f"Successfully built index with {len(documents)} documents")
            return True

        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise
            

    # Updated retrieve_context method for DataHandlerAgent class

    async def retrieve_context(self, query: str, k: int = 5) -> Dict:
        """
        Retrieve context with specific handling for different similarity thresholds.
        """
        try:
            # Check if vector store is initialized
            if self.vector_store is None:
                print("Warning: Vector store is not initialized. Attempting to load from default paths...")
                try:
                    # Try to load from default paths
                    self.load_index()
                    print("Successfully loaded index from default paths")
                except Exception as load_e:
                    print(f"Could not automatically load index: {str(load_e)}")
                    return {
                        'type': 'error',
                        'error': "Vector store not initialized. Please run 'financial-misinfo build' first or verify index paths."
                    }
            
            # Check if any documents were indexed
            if not self.chunks or not self.metadata:
                return {
                    'type': 'error',
                    'error': "No documents indexed. The system hasn't been properly initialized with data."
                }
            
            # Get query embedding and search
            query_embedding = self.embedding_model.encode([query])
            D, I = self.vector_store.search(np.array(query_embedding), k)
            
            # Make sure we have valid results
            if len(I) == 0 or len(I[0]) == 0:
                return {
                    'type': 'llm_fallback',
                    'similarity': 0.0,
                    'metadata': {'claim_text': query, 'evidence_sentence': 'No matching evidence found.'},
                    'context': query
                }
                
            # Get best match
            best_similarity = 1 / (1 + D[0][0])
            best_idx = I[0][0]
            
            # Make sure the index is within bounds
            if best_idx >= len(self.metadata) or best_idx >= len(self.chunks):
                return {
                    'type': 'error',
                    'error': f"Index out of bounds. Retrieved index {best_idx} but metadata has {len(self.metadata)} items and chunks has {len(self.chunks)} items."
                }
            
            # Handle different similarity thresholds
            if best_similarity >= 0.6:
                # High confidence match
                return {
                    'type': 'high_confidence',
                    'similarity': best_similarity,
                    'metadata': self.metadata[best_idx],
                    'context': self.chunks[best_idx]
                }
            else:
                return {
                    'type': 'llm_fallback',
                    'similarity': best_similarity,
                    'metadata': self.metadata[best_idx],
                    'context': self.chunks[best_idx]
                }

        except Exception as e:
            print(f"Error in retrieve_context: {str(e)}")
            return {
                'type': 'error',
                'error': str(e)
            }

    
    
    async def load_json_file(self, file_path: str):
        """Load a JSON file consistently, handling both array and line-by-line formats."""
        try:
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    print(f"Successfully loaded {len(data)} records from {file_path}")
                    return data
                except json.JSONDecodeError:
                    file.seek(0)
                    data = []
                    for line in file:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Error parsing line: {e}")
                    print(f"Successfully loaded {len(data)} records from {file_path}")
                    return data
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return []


    def _create_ivf_index(self, dimension: int, embeddings: np.ndarray):
        n_clusters = min(int(np.sqrt(len(embeddings))), 16384)
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_L2)
        index.train(embeddings)
        return index


    """
    def save_index(self):
        try:
            faiss.write_index(self.vector_store, "faiss_index.bin")
            with open("metadata.pkl", "wb") as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'chunks': self.chunks,
                    
                }, f)
            print("Index and metadata saved successfully")
        except Exception as e:
            print(f"Warning: Could not save index: {str(e)}")
    
    """


    def resolve_path(self, path):
        """
        Resolve a file path by checking multiple locations in order:
        1. If absolute path and exists, use it
        2. Current working directory
        3. Package data directory
        4. User's ~/.financial-misinfo/ directory
        
        Args:
            path: The file path to resolve
            
        Returns:
            Resolved absolute path as string
        """
        if not path:
            return None
        
        # Convert to Path object for easier handling
        path_obj = Path(path)
        
        # Extract just the filename if a full path was provided
        filename = path_obj.name
        
        # Check 1: If absolute path and file exists, use it
        if path_obj.is_absolute() and path_obj.exists():
            return str(path_obj)
        
        # Check 2: Check in current working directory
        cwd_path = Path.cwd() / filename
        if cwd_path.exists():
            return str(cwd_path)
        
        # Check 3: Check in package data directory
        try:
            import financial_misinfo
            pkg_data_dir = Path(os.path.dirname(financial_misinfo.__file__)) / "data"
            pkg_path = pkg_data_dir / filename
            if pkg_path.exists():
                return str(pkg_path)
        except (ImportError, AttributeError):
            pass
        
        # Check 4: Check in user's home .financial-misinfo directory
        user_dir = Path(os.path.expanduser("~")) / ".financial-misinfo"
        user_path = user_dir / filename
        if user_path.exists():
            return str(user_path)
        
        # For saving: If we're given a path with a directory, use it
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path
        
        # Default: Use the user directory path
        user_dir.mkdir(parents=True, exist_ok=True)
        return str(user_path)


    def save_index(self, vector_path=None, metadata_path=None):
        """
        Save the vector store and metadata to files with proper path resolution.
        
        Args:
            vector_path: Path to save the vector store (optional)
            metadata_path: Path to save the metadata (optional)
            
        Returns:
            Dict with the resolved paths used, or None on error
        """
        try:
            # Use provided paths or default to simple filenames
            vector_path = vector_path or "faiss_index.bin"
            metadata_path = metadata_path or "metadata.pkl"
            
            # Resolve paths
            resolved_vector_path = self.resolve_path(vector_path)
            resolved_metadata_path = self.resolve_path(metadata_path)
            
            # Save the files
            os.makedirs(os.path.dirname(resolved_vector_path), exist_ok=True)
            faiss.write_index(self.vector_store, resolved_vector_path)
            
            os.makedirs(os.path.dirname(resolved_metadata_path), exist_ok=True)
            with open(resolved_metadata_path, "wb") as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'chunks': self.chunks,
                }, f)
            
            print(f"Index saved to {resolved_vector_path}")
            print(f"Metadata saved to {resolved_metadata_path}")
            
            return {
                'vector_path': resolved_vector_path,
                'metadata_path': resolved_metadata_path
            }
        except Exception as e:
            print(f"Warning: Could not save index: {str(e)}")
            return None

    def load_index(self, vector_path=None, metadata_path=None):
        """
        Load the vector store and metadata from files with proper path resolution.
        
        Args:
            vector_path: Path to the vector store file (optional)
            metadata_path: Path to the metadata file (optional)
            
        Returns:
            Dict with the resolved paths used on success
            
        Raises:
            FileNotFoundError: If files cannot be found
            ValueError: If metadata format is invalid
        """
        try:
            # Use provided paths or default to simple filenames
            vector_path = vector_path or "faiss_index.bin"
            metadata_path = metadata_path or "metadata.pkl"
            
            # Resolve paths
            resolved_vector_path = self.resolve_path(vector_path)
            resolved_metadata_path = self.resolve_path(metadata_path)
            
            # Check if files exist
            if not os.path.exists(resolved_vector_path):
                raise FileNotFoundError(f"Vector store file not found at {resolved_vector_path} (original path: {vector_path})")
            
            if not os.path.exists(resolved_metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {resolved_metadata_path} (original path: {metadata_path})")
            
            # Load vector store
            self.vector_store = faiss.read_index(resolved_vector_path)
            
            # Load metadata
            with open(resolved_metadata_path, "rb") as f:
                data = pickle.load(f)
                
                # Validate data structure
                if not isinstance(data, dict):
                    raise ValueError(f"Invalid metadata format: expected dict, got {type(data)}")
                
                if 'metadata' not in data or 'chunks' not in data:
                    raise ValueError("Invalid metadata format: missing required keys")
                
                self.metadata = data['metadata']
                self.chunks = data['chunks']
            
            # Validate loaded data
            if len(self.metadata) == 0 or len(self.chunks) == 0:
                raise ValueError("Loaded empty metadata or chunks - index may be corrupted")
            
            print(f"Successfully loaded index from {resolved_vector_path} and metadata from {resolved_metadata_path}")
            
            # Mark that we've successfully loaded the index
            self._tried_autoload = True
            
            return {
                'vector_path': resolved_vector_path,
                'metadata_path': resolved_metadata_path
            }
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            # Clear partial state on error
            self.vector_store = None
            self.metadata = []
            self.chunks = []
            raise