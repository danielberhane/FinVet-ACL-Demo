
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


class VotingAgent:

    def vote(self, results: List[Dict]) -> Dict:
        """
        Enhanced voting function that determines the final verdict based on pipeline results.
        """
        # print("Initializing voting...")

        # Initialize default pipeline-specific results
        rag_A_result = {'label': 'unknown', 'evidence': 'No evidence provided', 'source': [], 'confidence': 0.0}
        rag_B_result = {'label': 'unknown', 'evidence': 'No evidence provided', 'source': [], 'confidence': 0.0}
        fact_check_result = {'label': 'unknown', 'evidence': 'No evidence provided', 'source': [], 'confidence': 0.0}

        # Default final verdict for cases with no valid results
        default_verdict = {
            'label': 'nei',
            'evidence': 'Not enough information',
            'source': [],
            'confidence': 0.0
        }

        # If there are no results, return with defaults
        if not results:
            # print("================ NO RESULTS CASE ================")
            return {
                'final_verdict': default_verdict,
                'rag_A': rag_A_result,
                'rag_B': rag_B_result,
                'fact_check': fact_check_result
            }

        # Extract pipeline results
        fact_check = next((r for r in results if r['pipeline'] == 'fact_check'), None)
        rag_A = next((r for r in results if r['pipeline'] == 'rag_A'), None)
        rag_B = next((r for r in results if r['pipeline'] == 'rag_B'), None)

        # Update individual pipeline results if available
        if rag_A:
            rag_A_result = {
                'label': rag_A.get('label', 'unknown'),
                'evidence': rag_A.get('evidence', 'No evidence provided'),
                'source': rag_A.get('source', []),
                'confidence': rag_A.get('confidence', 0.0)
            }

        if rag_B:
            rag_B_result = {
                'label': rag_B.get('label', 'unknown'),
                'evidence': rag_B.get('evidence', 'No evidence provided'),
                'source': rag_B.get('source', []),
                'confidence': rag_B.get('confidence', 0.0)
            }

        if fact_check:
            fact_check_result = {
                'label': fact_check.get('label', 'unknown'),
                'evidence': fact_check.get('evidence', 'No evidence provided'),
                'source': fact_check.get('source', []),
                'confidence': fact_check.get('confidence', 0.0)
            }

        # Case 1: Fact-check has confidence of exactly 1.0
        if fact_check and fact_check.get('confidence', 0.0) == 1.0:
            # print("================ CASE 1: FACT CHECK HAS CONFIDENCE 1.0 ================")
            return {
                'final_verdict': {
                    'label': fact_check['label'],
                    'evidence': fact_check['evidence'],
                    'source': fact_check.get('source', []),
                    'confidence': fact_check['confidence']
                },
                'rag_A': rag_A_result,
                'rag_B': rag_B_result,
                'fact_check': fact_check_result
            }

        # Check if all confidences are zero
        all_zero_confidence = all(r.get('confidence', 0.0) == 0.0 for r in results)
        if all_zero_confidence:
            # print("================ CASE 2: ALL ZERO CONFIDENCE ================")
            return {
                'final_verdict': default_verdict,
                'rag_A': rag_A_result,
                'rag_B': rag_B_result,
                'fact_check': fact_check_result
            }

        # Sort results by confidence score (highest first)
        sorted_results = sorted(results, key=lambda x: x.get('confidence', 0.0), reverse=True)

        # Get highest confidence result
        highest_confidence_result = sorted_results[0]
        highest_confidence = highest_confidence_result.get('confidence', 0.0)

        # Find all results with the same highest confidence
        tied_results = [r for r in sorted_results if r.get('confidence', 0.0) == highest_confidence]

        # Case 3: Single result with highest confidence
        if len(tied_results) == 1:
            # print("================ CASE 3: SINGLE HIGHEST CONFIDENCE ================")
            return {
                'final_verdict': {
                    'label': highest_confidence_result.get('label', 'nei'),
                    'evidence': highest_confidence_result.get('evidence', 'No evidence provided'),
                    'source': highest_confidence_result.get('source', []),
                    'confidence': highest_confidence
                },
                'rag_A': rag_A_result,
                'rag_B': rag_B_result,
                'fact_check': fact_check_result
            }

        # Case 4: Multiple results with equal highest confidence
        # print("================ CASE 4: MULTIPLE HIGHEST CONFIDENCE RESULTS ================")
        # Select the one with longest evidence
        selected = max(tied_results, key=lambda x: len(x.get('evidence', '')))
        return {
            'final_verdict': {
                'label': selected.get('label', 'nei'),
                'evidence': selected.get('evidence', 'No evidence provided'),
                'source': selected.get('source', []),
                'confidence': selected.get('confidence', 0.0)
            },
            'rag_A': rag_A_result,
            'rag_B': rag_B_result,
            'fact_check': fact_check_result
        }