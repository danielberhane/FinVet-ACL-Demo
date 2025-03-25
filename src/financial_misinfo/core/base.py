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


"""Base classes for the financial misinformation system."""

from typing import Dict

class LLMResponseParser:
    """Base class for parsing LLM responses"""

    def _parse_llm_response(self, text: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Extract only the response part after instruction
            if '[/INST]' in text:
                text = text.split('[/INST]')[-1].strip()

            # Initialize with default values
            result = {
                'label': 'nei',
                'evidence': 'Not enough information',
                'source': [],  # List to hold sources
                'confidence': 0.3
            }

            # Parse response line by line
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in lines:
                # Parse label
                if line.lower().startswith('label:'):
                    label_text = line.split(':', 1)[1].strip().lower()
                    if 'true' in label_text:
                        result['label'] = 'true'
                    elif 'false' in label_text:
                        result['label'] = 'false'
                    elif 'nei' in label_text or 'not enough' in label_text:
                        result['label'] = 'nei'

                # Parse evidence
                elif line.lower().startswith('evidence:'):
                    evidence_text = line.split(':', 1)[1].strip()
                    if evidence_text:
                        result['evidence'] = evidence_text

                # Parse source
                elif line.lower().startswith('source:'):
                    source_text = line.split(':', 1)[1].strip()
                    if source_text and source_text.lower() not in ['none', 'n/a']:
                        result['source'] = [source_text]  # Add source as list item

                # Parse confidence
                elif line.lower().startswith('confidence:'):
                    try:
                        confidence_text = line.split(':', 1)[1].strip()
                        import re
                        confidence_match = re.search(r'(\d+\.\d+|\d+)', confidence_text)
                        if confidence_match:
                            conf = float(confidence_match.group(1))
                            result['confidence'] = min(0.4, max(0.0, conf))
                    except Exception as e:
                        print(f"Error parsing confidence: {str(e)}")

            return result

        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return {
                'label': 'nei',
                'evidence': f'Error parsing response: {str(e)}',
                'source': [],
                'confidence': 0.0
            }