
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

# Missing imports

from financial_misinfo.core.base import LLMResponseParser

class FactCheckPipeline (LLMResponseParser):
  
    """Implements the Google Fact Check + LLM pipeline"""
    def __init__(self, model:str, google_api_key: str, hf_token: str):

        if not hf_token or hf_token.strip() == "":
            raise ValueError("HuggingFace token is missing or empty")

        # print("Initializing FactCheckPipeline...")
        
        self.model_name = model
        self.google_api_key = google_api_key
        self.hf_token = hf_token
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        
        
    """   
    async def process(self, claim: str) -> Dict:
  
        try:
            # print("\n----- Step 1: Querying Google Fact Check API -----")
            fact_check_result = await self.google_fact_check(claim)

            if fact_check_result and fact_check_result.get('claims') and len(fact_check_result['claims']) > 0:
                first_claim = fact_check_result['claims'][0]
                claim_review = first_claim.get('claimReview', [{}])[0]

                # Extract fields
                # Use textualRating or publisher's statement rather than just the title
                evidence = claim_review.get('textualRating', '')
                if not evidence:
                    evidence = claim_review.get('publisherName', '') + ': ' + claim_review.get('title', '')
                if not evidence or evidence == ': ':
                    evidence = 'No detailed evidence provided'
                raw_label = claim_review.get('textualRating', 'nei').lower()
                source = claim_review.get('url', '')
                confidence = 1.0 if fact_check_result.get('claims') else 0.0

                # Normalize labels
                if raw_label in ['true', 'mostly true', 'correct attribution', 'accurate', 'correct']:
                    label = 'true'
                elif raw_label in ['false', 'mostly false', 'scam', 'fake', 'labeled satire', 
                         'pants on fire', 'misleading', 'miscaptioned', 'distorts the facts', 
                         'exaggerates', 'spins the facts', 'not legit', 'legend', 'misattributed', 
                         'all bull']:
                    label = 'false'
                elif raw_label in ['mixture', 'unverified', 'unproven', 'half true', 'half']:
                    label = 'nei'
                else:
                    label = 'nei'

                return {
                    'pipeline': 'fact_check',
                    'label': label,
                    'evidence': evidence,
                    'source': source,
                    'confidence': confidence
                }

            
            else:
                
                #print("\n----- No Fact Check Results Found -----")
                #print("Falling back to LLM analysis...")
                return await self.llm_analyze(claim)

        except Exception as e:
            print(f"Error processing fact check: {str(e)}")
            return {
                'pipeline': 'fact_check',
                'label': 'nei',
                'evidence': f'Error processing fact check: {str(e)}',
                'source': '',
                'confidence': 0.0
            }
        
    
    """
    async def process(self, claim: str) -> Dict:
        """Process a claim using Google Fact Check results"""
        try:
            fact_check_result = await self.google_fact_check(claim)
            if fact_check_result and fact_check_result.get('claims') and len(fact_check_result['claims']) > 0:
                first_claim = fact_check_result['claims'][0]
                claim_review = first_claim.get('claimReview', [{}])[0]

                # Extract the raw label first for normalization
                raw_label = claim_review.get('textualRating', 'nei').lower()

                # Extract more detailed evidence with fallback options
                evidence = ""

                # Try getting reviewer's detailed explanation from different possible fields
                if 'reviewRating' in claim_review and isinstance(claim_review['reviewRating'], dict):
                    rating_obj = claim_review['reviewRating']
                    if 'alternateName' in rating_obj and rating_obj['alternateName']:
                        evidence = rating_obj['alternateName']

                # If no evidence yet, try getting publisher's conclusion/summary
                if not evidence and 'publisher' in claim_review and isinstance(claim_review['publisher'], dict):
                    publisher_name = claim_review['publisher'].get('name', '')
                    if publisher_name and raw_label:
                        evidence = f"{publisher_name} rated this claim as: {raw_label}"

                # Next try to get the textualRating with context
                if not evidence and raw_label:
                    evidence = f"Fact check conclusion: {raw_label}"

                # If still no evidence, try the title with context
                if not evidence and 'title' in claim_review and claim_review['title']:
                    title = claim_review['title']
                    evidence = f"Fact check summary: {title}"

                # Finally check if there's a description field
                if not evidence and 'description' in claim_review and claim_review['description']:
                    evidence = claim_review['description']

                # If all else fails, provide a generic message
                if not evidence:
                    evidence = "A fact check was found, but detailed conclusion is not available in the response."

                # Get source URL
                source = claim_review.get('url', '')
                confidence = 1.0 if fact_check_result.get('claims') else 0.0

                # Normalize labels
                if raw_label in ['true', 'mostly true', 'correct attribution', 'accurate', 'correct']:
                    label = 'true'
                elif raw_label in ['false', 'mostly false', 'scam', 'fake', 'labeled satire', 
                         'pants on fire', 'misleading', 'miscaptioned', 'distorts the facts', 
                         'exaggerates', 'spins the facts', 'not legit', 'legend', 'misattributed', 
                         'all bull']:
                    label = 'false'
                elif raw_label in ['mixture', 'unverified', 'unproven', 'half true', 'half']:
                    label = 'nei'
                else:
                    label = 'nei'

                return {
                    'pipeline': 'fact_check',
                    'label': label,
                    'evidence': evidence,
                    'source': source,
                    'confidence': confidence
                }

            else:
                return await self.llm_analyze(claim)
        except Exception as e:
            print(f"Error processing fact check: {str(e)}")
            return {
                'pipeline': 'fact_check',
                'label': 'nei',
                'evidence': f'Error processing fact check: {str(e)}',
                'source': '',
                'confidence': 0.0
            }
     

    async def llm_analyze(self, claim: str) -> Dict:
        """Direct LLM analysis when no Google fact check results available"""
        try:
            prompt = f"""<s>[INST]
    You are a collaborative team of expert fact-checkers consisting of:

    1. Financial Analysis & Economic Policy Expert:
       - Specialized in analyzing economic claims, financial markets, and monetary policy
       - Deep knowledge of corporate finance and market trends

    2. Political Misinformation & Election Integrity Specialist:
       - Expert in detecting political misinformation and disinformation
       - Specialized knowledge of election systems and voting processes
       - Deep understanding of political campaigns and messaging

    3. Government Policy Analyst:
       - Expertise in legislative processes and policy implementation
       - Deep knowledge of federal, state, and local government operations
       - Specialized in analyzing policy claims and their impacts

    4. Investigative Journalist:
       - Extensive experience in fact-checking and verification
       - Expertise in source evaluation and contextual analysis
       - Skilled in uncovering deceptive claims and narratives

    Carefully analyze the following claim for accuracy and verifiability.

    Claim: {claim}

    Analysis Framework:

    1. Core Claims Identification:
       - Break down and identify the main factual assertion(s)
       - Isolate verifiable components
       - Note any important context or qualifiers

    2. Step-by-Step Analysis:
       - Systematically evaluate each component of the claim
       - Cross-reference with known facts and reliable data
       - Consider historical and contextual factors
       - Apply relevant expertise from each team member

    3. Evidence Assessment:
       - Evaluate the reliability of available information
       - Identify any gaps in the evidence
       - Consider potential biases or misleading elements

    4. Final Determination:
       - Based on your collective analysis, label the claim as one of:
           * True: The core factual assertions of the claim are accurate.
           * False: The core factual assertions are demonstrably untrue or unsupported by credible evidence.
           * NEI (Not Enough Information): There is insufficient reliable data for a confident verification.

    Additionally, please provide:
    - Evidence: A detailed, evidence-based justification for your label, highlighting key facts, data, and reasoning from the relevant expert perspectives.
    - Source: A supporting reference or note if none is available.
    - Confidence: A numerical score between 0 and 1 indicating how confident you are in your label, evidence, and source.

    Return your response exactly in the following format (and no additional commentary):

    Label: [True/False/NEI]
    Evidence: [Your detailed explanation]
    Source: [Supporting reference or note]
    Confidence: [Score between 0 and 1]

    [/INST]</s>"""

            # Get response from LLM
            response = await self._make_llm_request(prompt)

            # print("=============================================== OUTPUT FROM LLM ============================================================")
            # print ("The response is ... ")
            # print(response)
            # print ("The response ends ")
            # print("============================================================================================================================")

            # Parse the LLM response
            result = self._parse_llm_response(response)
            # print(f"Parsed result: {result}")

            # Add pipeline field
            result['pipeline'] = 'fact_check'

            return result

        except Exception as e:
            print(f"Error in llm_analyze: {str(e)}")
            return {
                'pipeline': 'fact_check',
                'label': 'nei',
                'evidence': f'Error in LLM analysis: {str(e)}',
                'source': '',
                'confidence': 0.0
            }

    
   
        
    async def google_fact_check(self, claim: str) -> Dict:
        """Query Google Fact Check API with improved error handling"""


        try:
            # print("\n----- Step 1: Preparing API Request -----")
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {"query": claim, "key": self.google_api_key}
            # print(f"Request URL: {url}")
            # print(f"Request parameters prepared (key hidden)")

            # print("\n----- Step 2: Sending API Request -----")
            response = requests.get(url, params=params, timeout=10)
            # print(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                # print("\n----- Step 3: Processing Successful Response -----")
                result = response.json()
                # print("Response received and parsed")
                # print("Response structure:")
                # print(json.dumps(result, indent=2))

                # Additional response analysis
                # if 'claims' in result:
                #    print(f"\nNumber of claims found: {len(result['claims'])}")
                    # for idx, claim in enumerate(result['claims']):
                        # print(f"\nClaim {idx + 1}:")
                        # print(f"- Text: {claim.get('text', 'No text')[:100]}...")
                        # print(f"- Rating: {claim.get('textualRating', 'No rating')}")
                #else:
                    # print("\nNo claims found in response")

                 #rint("\n========== GOOGLE FACT CHECK FUNCTION END ==========")
                return result
            else:
                print(f"\n!!! API ERROR !!!")
                print(f"Status code: {response.status_code}")
                print(f"Response text: {response.text}")
                print("\n========== GOOGLE FACT CHECK FUNCTION END ==========")
                return {}

        except Exception as e:
            print(f"\n!!! ERROR IN GOOGLE FACT CHECK FUNCTION !!!")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print("Traceback:", traceback.format_exc())
            print("\n========== GOOGLE FACT CHECK FUNCTION END ==========")
            return {}

    
    
    
    async def _make_llm_request(self, prompt: str) -> str:
        """Make request to HuggingFace API with improved validation and error handling."""
        try:
            # Validate token again (in case it was changed after initialization)
            if not self.hf_token or self.hf_token.strip() == "":
                raise ValueError("HuggingFace token is missing or empty")

            # Debug statement to show token (partially masked for security)
            token_preview = self.hf_token[:4] + "..." + self.hf_token[-4:] if len(self.hf_token) > 8 else "[empty]"
                
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.1
                    }
                },
                timeout=30  # Add timeout to prevent hanging on slow responses
            )
            
            # Check for various error conditions
            if response.status_code == 401:
                raise ValueError(f"API authentication failed: Invalid HuggingFace token")
            elif response.status_code == 404:
                raise ValueError(f"Model not found: {self.model_name}")
            elif response.status_code == 503:
                raise ValueError(f"API server overloaded or unavailable")
            elif response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if not result or not isinstance(result, list) or not result[0]:
                raise Exception("Invalid API response format")
            
            return result[0].get('generated_text', '')

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error when calling HuggingFace API: {str(e)}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response from API: {response.text[:100]}...")
        except Exception as e:
            print(f"Error in LLM request: {str(e)}")
            raise
    