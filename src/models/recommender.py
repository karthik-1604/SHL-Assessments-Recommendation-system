#!/usr/bin/env python3
"""
SHL Recommender with Gemini API Embeddings, Smart Balance Logic, and Data-Driven Analysis
Complete fixed version with proper initialization and caching
"""

import json
import numpy as np
import faiss
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
import os
import pickle
import hashlib
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLRecommender:
    """Advanced SHL Recommender with Gemini API embeddings, smart balance, and caching"""
    
    def __init__(self):
        self.assessments = []
        self.assessment_embeddings = None
        self.faiss_index = None
        
        # Cache settings
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_cache_file = self.cache_dir / "assessment_embeddings.pkl"
        self.faiss_cache_file = self.cache_dir / "faiss_index.bin"
        self.query_cache = {}  # In-memory cache for query embeddings
        self.query_cache_file = self.cache_dir / "query_embeddings.pkl"
        
        # Gemini API setup
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.embedding_provider = 'gemini'  # Fixed to use gemini
        
        logger.info(f"🚀 Initializing SHL Recommender with {self.embedding_provider} embeddings")
        
        # Initialize Gemini
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            logger.info("✅ Gemini API configured")
        else:
            logger.error("❌ Gemini API key not found in environment")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Load query cache
        self._load_query_cache()
    
    def _load_query_cache(self):
        """Load query embeddings cache"""
        try:
            if self.query_cache_file.exists():
                with open(self.query_cache_file, 'rb') as f:
                    self.query_cache = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.query_cache)} cached query embeddings")
            else:
                logger.info("📝 Starting with empty query cache")
        except Exception as e:
            logger.error(f"❌ Failed to load query cache: {e}")
            self.query_cache = {}
    
    def _save_query_cache(self):
        """Save query embeddings cache"""
        try:
            with open(self.query_cache_file, 'wb') as f:
                pickle.dump(self.query_cache, f)
            logger.info(f"💾 Saved {len(self.query_cache)} query embeddings to cache")
        except Exception as e:
            logger.error(f"❌ Failed to save query cache: {e}")
    
    def _get_data_hash(self) -> str:
        """Get hash of current assessment data for cache validation"""
        data_str = json.dumps(self.assessments, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _load_cached_embeddings(self) -> bool:
        """Load cached embeddings and FAISS index if available"""
        try:
            if not self.embeddings_cache_file.exists() or not self.faiss_cache_file.exists():
                logger.info("📝 No cached embeddings found")
                return False
            
            # Load cached embeddings with metadata
            with open(self.embeddings_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache against current data
            current_hash = self._get_data_hash()
            if cache_data.get('data_hash') != current_hash:
                logger.info("🔄 Data changed, cache invalid")
                return False
            
            # Load embeddings
            self.assessment_embeddings = cache_data['embeddings']
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(self.faiss_cache_file))
            
            logger.info(f"✅ Loaded cached embeddings: {self.assessment_embeddings.shape}")
            logger.info(f"✅ Loaded cached FAISS index: {self.faiss_index.ntotal} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load cached embeddings: {e}")
            return False
    
    def _save_embeddings_cache(self):
        """Save embeddings and FAISS index to cache"""
        try:
            # Save embeddings with metadata
            cache_data = {
                'embeddings': self.assessment_embeddings,
                'data_hash': self._get_data_hash(),
                'timestamp': time.time(),
                'embedding_model': 'text-embedding-004'
            }
            
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, str(self.faiss_cache_file))
            
            logger.info("💾 Saved embeddings and FAISS index to cache")
            
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings cache: {e}")
    
    def _get_query_hash(self, query: str) -> str:
        """Get hash for query to use as cache key"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _analyze_assessment_patterns(self):
        """Analyze the actual assessment data to understand domains and categorization patterns"""
        
        # Extract all unique test types and their frequencies
        self.test_type_patterns = {}
        self.domain_keywords = {}
        self.assessment_categories = {
            'technical': [],
            'behavioral': [],
            'mixed': [],
            'domain_specific': {}
        }
        
        logger.info("🔍 Analyzing assessment patterns from actual data...")
        
        for assessment in self.assessments:
            name = assessment.get('name', '').lower()
            test_types = assessment.get('test_type', [])
            description = assessment.get('description', '').lower()
            
            # Track test type patterns
            for test_type in test_types:
                if test_type not in self.test_type_patterns:
                    self.test_type_patterns[test_type] = []
                self.test_type_patterns[test_type].append(name)
            
            # Professional domains  
            domain_indicators = {
                'programming': ['java', 'python', 'javascript', 'c++', 'programming', 'coding'],
                'web_development': ['html', 'css', 'react', 'angular', 'web', 'frontend'],
                'database': ['sql', 'oracle', 'mongodb', 'database'],
                'cloud_devops': ['aws', 'docker', 'kubernetes', 'jenkins', 'cloud'],
                'engineering': ['mechanical', 'civil', 'electrical', 'aerospace', 'engineering'],
                'healthcare': ['nursing', 'medical', 'pharmacology', 'healthcare', 'clinical'],
                'finance': ['accounting', 'financial', 'banking', 'finance'],
                'science': ['chemistry', 'biology', 'physics', 'biochemistry'],
                'business': ['marketing', 'operations', 'management', 'business'],
                'design': ['photoshop', 'design', 'creative'],
                'data': ['data science', 'statistics', 'analytics', 'tableau']
            }
            
            # Categorize by domain
            for domain, keywords in domain_indicators.items():
                if any(keyword in name or keyword in description for keyword in keywords):
                    if domain not in self.assessment_categories['domain_specific']:
                        self.assessment_categories['domain_specific'][domain] = []
                    self.assessment_categories['domain_specific'][domain].append(assessment)
                    break
        
        # Categorize by test type patterns
        for assessment in self.assessments:
            test_types = assessment.get('test_type', [])
            test_types_str = ' '.join(test_types).lower()
            
            has_technical = any(word in test_types_str for word in 
                              ['knowledge', 'skills', 'ability', 'aptitude', 'simulations'])
            has_behavioral = any(word in test_types_str for word in 
                               ['personality', 'behavior', 'behaviour', 'competencies'])
            
            if has_technical and has_behavioral:
                self.assessment_categories['mixed'].append(assessment)
            elif has_technical:
                self.assessment_categories['technical'].append(assessment)
            elif has_behavioral:
                self.assessment_categories['behavioral'].append(assessment)
        
        logger.info(f"📊 Analysis complete:")
        logger.info(f"   - Technical assessments: {len(self.assessment_categories['technical'])}")
        logger.info(f"   - Behavioral assessments: {len(self.assessment_categories['behavioral'])}")
        logger.info(f"   - Mixed assessments: {len(self.assessment_categories['mixed'])}")
        logger.info(f"   - Domain-specific categories: {len(self.assessment_categories['domain_specific'])}")
        
        return True
    
    def load_shl_data(self, data_path: str = "data/raw/shl_assessments_377_complete.json"):
        """Load SHL assessments data"""
        try:
            logger.info(f"📊 Loading SHL data from {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.assessments)} SHL assessments")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return False
    
    def get_gemini_embeddings(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """Get embeddings from Gemini API in batches"""
        
        all_embeddings = []
        
        logger.info(f"🧠 Getting Gemini embeddings for {len(texts)} texts")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Process each text individually to avoid batch issues
                batch_embeddings = []
                
                for text in batch:
                    # Use Gemini embedding model
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document"
                    )
                    
                    embedding = np.array(result['embedding'])
                    batch_embeddings.append(embedding)
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                
                batch_embeddings = np.array(batch_embeddings)
                all_embeddings.append(batch_embeddings)
                
                logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Rate limiting between batches
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"❌ Gemini embedding failed for batch {i}: {e}")
                # Fallback to random embeddings for failed batch
                fallback_embeddings = np.random.randn(len(batch), 768) * 0.1
                all_embeddings.append(fallback_embeddings)
        
        if not all_embeddings:
            logger.error("❌ No embeddings generated")
            return np.random.randn(len(texts), 768) * 0.1
            
        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"✅ Generated {final_embeddings.shape[0]} embeddings of dimension {final_embeddings.shape[1]}")
        
        return final_embeddings
    
    def create_assessment_texts(self):
        """Create enhanced searchable text from assessment data"""
        texts = []
        
        for assessment in self.assessments:
            text_parts = []
            
            # Assessment name (high importance)
            if assessment.get('name'):
                name = assessment['name']
                text_parts.append(name)
                text_parts.append(name)  # Duplicate for weight
            
            # Test types (critical for balance)
            if assessment.get('test_type'):
                for test_type in assessment['test_type']:
                    text_parts.append(test_type)
                    text_parts.append(test_type)  # Duplicate for weight
            
            # Description
            if assessment.get('description'):
                text_parts.append(assessment['description'])
            
            # Duration and other metadata
            if assessment.get('duration'):
                text_parts.append(f"duration {assessment['duration']} minutes")
            
            combined_text = ' '.join(text_parts)
            texts.append(combined_text)
        
        return texts
    
    def build_vector_index(self):
        """Build FAISS vector index with Gemini embeddings (using cache when possible)"""
        try:
            logger.info("🔧 Building vector index with caching...")
            start_time = time.time()
            
            # Try to load from cache first
            if self._load_cached_embeddings():
                build_time = time.time() - start_time
                logger.info(f"⚡ Vector index loaded from cache in {build_time:.2f} seconds")
                return True
            
            # Cache miss - generate embeddings
            logger.info("🧠 Cache miss - generating new embeddings...")
            
            # Create assessment texts
            texts = self.create_assessment_texts()
            
            # Generate embeddings with Gemini
            embeddings = self.get_gemini_embeddings(texts)
            
            # Store embeddings
            self.assessment_embeddings = embeddings
            
            # Build FAISS index
            logger.info("🔍 Building FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            
            # Normalize embeddings for cosine similarity
            if embeddings.size > 0:
                normalized_embeddings = embeddings.copy().astype(np.float32)
                normalized_embeddings = np.ascontiguousarray(normalized_embeddings)
                faiss.normalize_L2(normalized_embeddings)
                self.faiss_index.add(normalized_embeddings)
            else:
                logger.error("❌ Empty embeddings array")
                return False
            
            # Save to cache
            self._save_embeddings_cache()
            
            build_time = time.time() - start_time
            logger.info(f"✅ Vector index built and cached in {build_time:.2f} seconds")
            logger.info(f"📏 Index dimension: {embeddings.shape[1]}")
            logger.info(f"📊 Indexed {len(self.assessments)} assessments")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector index building failed: {e}")
            return False
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get single query embedding using Gemini (with caching)"""
        
        # Check cache first
        query_hash = self._get_query_hash(query)
        if query_hash in self.query_cache:
            logger.info("⚡ Using cached query embedding")
            return self.query_cache[query_hash]
        
        try:
            logger.info("🧠 Generating new query embedding")
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            
            embedding = np.array(result['embedding'], dtype=np.float32).reshape(1, -1)
            
            # Cache the embedding
            self.query_cache[query_hash] = embedding
            
            # Periodically save cache (every 10 new queries)
            if len(self.query_cache) % 10 == 0:
                self._save_query_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Query embedding failed: {e}")
            # Return random embedding as fallback with correct dtype
            dim = self.assessment_embeddings.shape[1] if self.assessment_embeddings is not None else 768
            embedding = np.random.randn(1, dim).astype(np.float32) * 0.1
            
            # Don't cache failed embeddings
            return embedding
    
    def analyze_query_intent(self, query: str) -> Dict:
        """Analyze query intent using Gemini API with improved balance detection"""
        
        try:
            # Enhanced prompt for better balance detection
            prompt = f"""
            Analyze this job/role query and determine what types of assessments are needed:
            
            Query: "{query}"
            
            Return a JSON response with:
            1. technical_skills: List of specific technical skills mentioned
            2. soft_skills: List of behavioral/soft skills mentioned  
            3. domain: The job domain (software, finance, healthcare, etc.)
            4. needs_technical: true/false if technical assessments needed
            5. needs_behavioral: true/false if behavioral assessments needed
            6. balance_required: true if BOTH technical AND behavioral skills are important
            7. priority: "technical", "behavioral", or "balanced"
            
            Important rules:
            - If query mentions BOTH technical skills AND collaboration/communication/teamwork = balance_required: true
            - If it's a management/leadership role = needs_behavioral: true
            - If it's a technical role (developer, engineer, analyst) = needs_technical: true
            - If unsure, default to balanced approach
            
            Return only valid JSON.
            """
            
            # Use the correct Gemini API method
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            intent = json.loads(response_text)
            logger.info(f"🎯 Query intent: {intent.get('priority', 'unknown')} priority, balance: {intent.get('balance_required', False)}")
            
            return intent
            
        except Exception as e:
            logger.error(f"❌ Gemini intent analysis failed: {e}")
            return self._fallback_intent_analysis(query)
    
    def _fallback_intent_analysis(self, query: str) -> Dict:
        """Enhanced fallback rule-based intent analysis using actual assessment data"""
        
        query_lower = query.lower()
        
        # Dynamic domain detection based on actual assessment data
        detected_domains = []
        technical_skills = []
        soft_skills = []
        
        # Check against actual domain categories from our data
        if hasattr(self, 'assessment_categories'):
            for domain, assessments in self.assessment_categories.get('domain_specific', {}).items():
                # Extract keywords from actual assessment names in this domain
                domain_keywords = set()
                for assessment in assessments[:10]:  # Sample first 10 assessments
                    name_words = assessment.get('name', '').lower().split()
                    domain_keywords.update([word for word in name_words if len(word) > 3])
                
                # Check if query matches this domain
                if any(keyword in query_lower for keyword in domain_keywords):
                    detected_domains.append(domain)
                    # Extract the specific technical skills mentioned
                    technical_skills.extend([keyword for keyword in domain_keywords if keyword in query_lower])
        
        # Universal behavioral skill detection
        behavioral_keywords = {
            'communication': ['communication', 'communicate', 'verbal', 'written', 'presentation'],
            'collaboration': ['collaborate', 'collaboration', 'teamwork', 'team work', 'cross-functional'],
            'leadership': ['leadership', 'lead', 'management', 'supervise', 'mentor', 'coaching'],
            'interpersonal': ['interpersonal', 'social', 'relationship', 'networking'],
            'customer_focus': ['customer', 'client', 'service', 'satisfaction'],
            'problem_solving': ['problem solving', 'analytical', 'critical thinking', 'troubleshoot'],
            'adaptability': ['adaptable', 'flexible', 'change management', 'agile'],
            'business_acumen': ['business', 'commercial', 'stakeholder', 'business teams']
        }
        
        for behavior_type, keywords in behavioral_keywords.items():
            found_behaviors = [skill for skill in keywords if skill in query_lower]
            if found_behaviors:
                soft_skills.extend(found_behaviors)
        
        # Enhanced balance detection
        balance_required = False
        
        # Strong balance indicators
        balance_patterns = [
            r'who can also', r'and also', r'both .* and', r'technical and',
            r'programming and communication', r'development and collaboration',
            r'effectively with.*team', r'collaborate.*with.*business'
        ]
        
        for pattern in balance_patterns:
            if re.search(pattern, query_lower):
                balance_required = True
                break
        
        # Role + soft skill combination
        technical_roles = ['developer', 'engineer', 'programmer', 'analyst', 'architect']
        collaboration_indicators = ['collaborate', 'team', 'communication', 'business teams', 'stakeholder']
        
        has_technical_role = any(role in query_lower for role in technical_roles)
        has_collaboration_need = any(indicator in query_lower for indicator in collaboration_indicators)
        
        if has_technical_role and has_collaboration_need:
            balance_required = True
        
        # Leadership roles automatically need both
        leadership_roles = ['manager', 'lead', 'director', 'supervisor', 'head', 'chief']
        if any(role in query_lower for role in leadership_roles):
            balance_required = True
            if 'leadership' not in soft_skills:
                soft_skills.append('leadership')
        
        # Determine primary domain
        primary_domain = detected_domains[0] if detected_domains else 'general'
        
        # Determine needs
        needs_technical = len(technical_skills) > 0 or has_technical_role or len(detected_domains) > 0
        needs_behavioral = len(soft_skills) > 0 or balance_required or any(role in query_lower for role in leadership_roles)
        
        # If both detected, balance is required
        if needs_technical and needs_behavioral:
            balance_required = True
        
        # Determine priority
        if balance_required:
            priority = 'balanced'
        elif needs_behavioral and not needs_technical:
            priority = 'behavioral'
        elif needs_technical and not needs_behavioral:
            priority = 'technical'
        else:
            priority = 'balanced'  # Default to balanced when unclear
        
        return {
            'technical_skills': technical_skills,
            'soft_skills': soft_skills,
            'detected_domains': detected_domains,
            'primary_domain': primary_domain,
            'needs_technical': needs_technical,
            'needs_behavioral': needs_behavioral,
            'balance_required': balance_required,
            'priority': priority,
            'confidence': 'high' if balance_required or len(technical_skills) > 0 else 'medium'
        }
    
    def search_assessments(self, query: str, k: int = 20) -> List[Dict]:
        """Search for relevant assessments"""
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        
        # Ensure proper dtype and normalization
        query_embedding = query_embedding.astype(np.float32)
        query_embedding = np.ascontiguousarray(query_embedding)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.assessments):
                result = self.assessments[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def apply_smart_balance_logic(self, results: List[Dict], intent: Dict) -> List[Dict]:
        """Apply intelligent data-driven balance logic"""
        
        if not intent.get('balance_required', False):
            logger.info(f"🎯 Single domain focus ({intent.get('priority', 'unknown')}) - returning top results")
            return results[:10]
        
        primary_domain = intent.get('primary_domain', 'general')
        detected_domains = intent.get('detected_domains', [])
        
        logger.info(f"🎯 Applying data-driven balance logic for {primary_domain} domain")
        
        # Dynamic assessment categorization based on actual patterns
        technical_assessments = []
        behavioral_assessments = []
        mixed_assessments = []
        domain_specific_assessments = []
        
        for result in results:
            assessment_name = result.get('name', '').lower()
            test_types = result.get('test_type', [])
            test_types_str = ' '.join(test_types).lower()
            description = result.get('description', '').lower()
            combined_text = f"{assessment_name} {test_types_str} {description}"
            
            # Check if this assessment matches detected domains
            is_domain_specific = False
            if detected_domains:
                for domain in detected_domains:
                    # Check against our analyzed domain categories
                    if hasattr(self, 'assessment_categories') and domain in self.assessment_categories.get('domain_specific', {}):
                        domain_assessments = self.assessment_categories['domain_specific'][domain]
                        if any(result.get('name', '') == da.get('name', '') for da in domain_assessments):
                            is_domain_specific = True
                            domain_specific_assessments.append(result)
                            break
            
            # Dynamic categorization based on test types
            is_technical = False
            is_behavioral = False
            
            # Technical indicators from our data analysis
            technical_indicators = [
                'knowledge', 'skills', 'ability', 'aptitude', 'simulations',
                'programming', 'development', 'technical', 'engineering'
            ]
            
            # Behavioral indicators from our data analysis  
            behavioral_indicators = [
                'personality', 'behavior', 'behaviour', 'competencies',
                'opq', 'leadership', 'communication', 'interpersonal',
                'motivation', 'emotional', 'team', 'management'
            ]
            
            # Check for technical patterns
            for indicator in technical_indicators:
                if indicator in combined_text:
                    is_technical = True
                    break
            
            # Check for behavioral patterns
            for indicator in behavioral_indicators:
                if indicator in combined_text:
                    is_behavioral = True
                    break
            
            # Special domain-specific rules based on actual data patterns
            # Java assessments are always technical for Java queries
            if primary_domain == 'programming' and 'java' in assessment_name:
                is_technical = True
                is_domain_specific = True
            
            # OPQ assessments are always behavioral
            if 'opq' in assessment_name or 'personality' in assessment_name:
                is_behavioral = True
            
            # Categorize the assessment
            if not is_domain_specific:
                if is_technical and is_behavioral:
                    mixed_assessments.append(result)
                elif is_technical:
                    technical_assessments.append(result)
                elif is_behavioral:
                    behavioral_assessments.append(result)
                else:
                    # Default categorization based on test types
                    if any(word in test_types_str for word in ['knowledge', 'skills', 'ability']):
                        technical_assessments.append(result)
                    elif any(word in test_types_str for word in ['personality', 'behavior', 'competencies']):
                        behavioral_assessments.append(result)
                    else:
                        technical_assessments.append(result)  # Default to technical
        
        # Smart balance strategy with domain prioritization
        balanced_results = []
        
        # Priority 1: Domain-specific assessments (highest relevance)
        domain_count = min(len(domain_specific_assessments), 4)
        balanced_results.extend(domain_specific_assessments[:domain_count])
        
        # Priority 2: Mixed assessments (satisfy both requirements)
        mixed_count = min(len(mixed_assessments), 2)
        balanced_results.extend(mixed_assessments[:mixed_count])
        
        # Calculate remaining slots for balance
        remaining_slots = 10 - len(balanced_results)
        
        # Ensure behavioral representation (at least 3-4 behavioral assessments)
        behavioral_target = max(2, remaining_slots // 2)
        behavioral_count = min(len(behavioral_assessments), behavioral_target)
        
        # Prioritize interpersonal/communication assessments for collaboration queries
        if 'collaborate' in intent.get('soft_skills', []) or 'communication' in intent.get('soft_skills', []):
            # Sort behavioral assessments to prioritize communication/interpersonal
            behavioral_assessments.sort(key=lambda x: (
                'interpersonal' in x.get('name', '').lower() or 
                'communication' in x.get('name', '').lower()
            ), reverse=True)
        
        balanced_results.extend(behavioral_assessments[:behavioral_count])
        
        # Fill remaining with technical assessments
        remaining_after_behavioral = 10 - len(balanced_results)
        if remaining_after_behavioral > 0:
            # Prefer domain-specific, then general technical
            remaining_technical = (
                domain_specific_assessments[domain_count:] + 
                technical_assessments
            )
            # Sort by similarity score
            remaining_technical.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            technical_count = min(len(remaining_technical), remaining_after_behavioral)
            balanced_results.extend(remaining_technical[:technical_count])
        
        # If still not enough, fill with any remaining assessments
        if len(balanced_results) < 10:
            remaining = []
            for assessment_list in [behavioral_assessments[behavioral_count:], 
                                  technical_assessments, mixed_assessments[mixed_count:]]:
                for assessment in assessment_list:
                    if assessment not in balanced_results:
                        remaining.append(assessment)
            
            remaining.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            needed = 10 - len(balanced_results)
            balanced_results.extend(remaining[:needed])
        
        # Analysis and logging
        final_technical = sum(1 for r in balanced_results 
                            if any(word in ' '.join(r.get('test_type', [])).lower() + ' ' + r.get('name', '').lower()
                                  for word in ['knowledge', 'skills', 'ability', 'technical', 'programming']))
        
        final_behavioral = sum(1 for r in balanced_results 
                             if any(word in ' '.join(r.get('test_type', [])).lower() + ' ' + r.get('name', '').lower()
                                   for word in ['personality', 'behavior', 'behaviour', 'competencies', 'opq', 'interpersonal']))
        
        final_domain_specific = len([r for r in balanced_results if r in domain_specific_assessments])
        
        logger.info(f"📊 Final balance: {final_technical} technical, {final_behavioral} behavioral, "
                   f"{final_domain_specific} domain-specific, {len(balanced_results)} total")
        logger.info(f"🎯 Primary domain: {primary_domain}")
        
        # Log actual assessment names for debugging
        logger.info(f"🔍 Selected assessments:")
        for i, assessment in enumerate(balanced_results[:5], 1):
            name = assessment.get('name', 'Unknown')
            types = assessment.get('test_type', [])
            logger.info(f"  {i}. {name} - {types}")
        
        return balanced_results[:10]
    
    def recommend(self, query: str, top_k: int = 10) -> Dict:
        """Main recommendation function with intelligent balance"""
        
        try:
            logger.info(f"🎯 Processing recommendation for: '{query}'")
            start_time = time.time()
            
            # Analyze query intent
            intent = self.analyze_query_intent(query)
            
            # Search for similar assessments
            results = self.search_assessments(query, k=20)
            
            # Apply intelligent balance logic
            final_results = self.apply_smart_balance_logic(results, intent)
            
            # Format for exact API response structure
            recommended_assessments = []
            for result in final_results:
                assessment = {
                    'url': result.get('url', ''),
                    'name': result.get('name', ''),
                    'adaptive_support': result.get('adaptive_support', 'No'),
                    'description': result.get('description', ''),
                    'duration': result.get('duration', 0),
                    'remote_support': result.get('remote_support', 'No'),
                    'test_type': result.get('test_type', [])
                }
                recommended_assessments.append(assessment)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Recommendation completed in {processing_time:.2f} seconds")
            logger.info(f"📊 Returning {len(recommended_assessments)} balanced recommendations")
            
            # Return in exact API format
            return {
                'recommended_assessments': recommended_assessments
            }
            
        except Exception as e:
            logger.error(f"❌ Recommendation failed: {e}")
            # Return fallback in correct format
            return {
                'recommended_assessments': [{
                    'url': 'https://www.shl.com/solutions/products/product-catalog/',
                    'name': 'SHL Product Catalog',
                    'adaptive_support': 'No',
                    'description': 'Browse the full SHL assessment catalog for suitable options.',
                    'duration': 0,
                    'remote_support': 'Yes',
                    'test_type': ['General']
                }]
            }
    
    def clear_cache(self):
        """Clear all cached embeddings and indexes"""
        try:
            if self.embeddings_cache_file.exists():
                self.embeddings_cache_file.unlink()
            if self.faiss_cache_file.exists():
                self.faiss_cache_file.unlink()
            if self.query_cache_file.exists():
                self.query_cache_file.unlink()
            
            self.query_cache = {}
            logger.info("🗑️ Cleared all caches")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
    
    def get_cache_info(self) -> Dict:
        """Get information about cache status"""
        return {
            'embeddings_cached': self.embeddings_cache_file.exists(),
            'faiss_cached': self.faiss_cache_file.exists(),
            'query_cache_size': len(self.query_cache),
            'cache_directory': str(self.cache_dir)
        }
    
    def initialize(self) -> bool:
        """Initialize the complete system with data analysis"""
        
        logger.info("🚀 Initializing SHL Recommender System")
        
        steps = [
            ("Loading SHL data", self.load_shl_data),
            ("Analyzing assessment patterns", self._analyze_assessment_patterns),
            ("Building vector index", self.build_vector_index)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"⏳ {step_name}...")
            if not step_func():
                logger.error(f"❌ Failed at: {step_name}")
                return False
        
        logger.info("✅ SHL Recommender System initialized successfully!")
        
        # Show cache status
        cache_info = self.get_cache_info()
        logger.info(f"💾 Cache status: embeddings={cache_info['embeddings_cached']}, "
                   f"faiss={cache_info['faiss_cached']}, query_cache={cache_info['query_cache_size']}")
        
        return True
    
    def __del__(self):
        """Cleanup - save query cache on destruction"""
        try:
            if hasattr(self, 'query_cache') and self.query_cache:
                self._save_query_cache()
        except:
            pass  # Ignore errors during cleanup


# Test the system
if __name__ == "__main__":
    
    recommender = SHLRecommender()
    
    if recommender.initialize():
        
        test_queries = [
            "Java developer with good communication skills",
            "Customer service representative", 
            "Senior software engineer who can lead teams",
            "Python programming assessment needed",
            "Accounting manager position",
            "Bank customer service role"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            result = recommender.recommend(query)
            assessments = result.get('recommended_assessments', [])
            
            for i, assessment in enumerate(assessments[:5], 1):
                print(f"{i}. {assessment['name']}")
                print(f"   Types: {assessment['test_type']}")
                print(f"   URL: {assessment['url']}")
                print()
    
    else:
        print("❌ Failed to initialize recommender system")