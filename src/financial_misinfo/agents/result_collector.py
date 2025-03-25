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

class ResultCollectorAgent:
    def __init__(self):
        # Maps text labels to numeric values
        self.label_mapping = {'true': 1, 'false': 0, 'nei': 2}

    def collect(self, results: List[Dict]) -> List[Dict]:
        """Normalize results while preserving original information"""
        normalized_results = []

        for result in results:
            if isinstance(result, Exception):
                continue

            try:
                # Create normalized result maintaining all original fields
                normalized_results.append({
                    'pipeline': result.get('pipeline', 'unknown'),
                    'label': result.get('label', 'nei').lower(),
                    'evidence':result.get('evidence', 'Not enough information'),
                    'confidence': float(result.get('confidence', 0.0)),
                    'source': result.get('source', [])
                })
            except Exception as e:
                print(f"Error normalizing result: {str(e)}")
                continue

        return normalized_results