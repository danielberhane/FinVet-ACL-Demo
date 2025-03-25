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

import torch
torch.set_num_threads(1)

from financial_misinfo.agents.orchestrator import OrchestratorAgent
from financial_misinfo.utils.config import load_config

class FinancialMisinfoSystem:
    """Financial Misinformation Detection System.
    
    This is the main entry point for using the system.
    """
    
    def __init__(self, hf_token: str, google_api_key: str, config: dict = None):
        # Validate API keys
        if not hf_token or not isinstance(hf_token, str) or hf_token.strip() == "":
            raise ValueError("HuggingFace token is required and must be a non-empty string")
        
        if not google_api_key or not isinstance(google_api_key, str) or google_api_key.strip() == "":
            raise ValueError("Google API key is required and must be a non-empty string")

        # If config not provided, load default
        if config is None:
            config = load_config(verbose=False)
        
        # Get model configurations
        models = config.get("models", {})
        primary_llm = models.get("primary_llm", "meta-llama/Llama-3.3-70B-Instruct")
        secondary_llm = models.get("secondary_llm", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        
        # Initialize the orchestrator with proper model configurations
        self.orchestrator = OrchestratorAgent(
            hf_token=hf_token,
            google_api_key=google_api_key,
            primary_model=primary_llm,
            secondary_model=secondary_llm
        )