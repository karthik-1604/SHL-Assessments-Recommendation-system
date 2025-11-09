#!/usr/bin/env python3
"""
Enhanced Smart Query Processor with Dynamic Domain Learning and Advanced JD Analysis
Integrates with SHL Assessment Data for Intelligent Processing
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Try to download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SkillMention:
    """Represents a detected skill with context"""
    skill: str
    context: str
    priority: str  # 'required', 'preferred', 'nice_to_have'
    section: str   # 'responsibilities', 'requirements', 'qualifications'
    confidence: float

@dataclass
class QueryIntent:
    """Enhanced query intent with detailed analysis"""
    # Basic skills
    technical_skills: List[SkillMention] = field(default_factory=list)
    behavioral_skills: List[SkillMention] = field(default_factory=list)
    
    # Context analysis
    experience_level: str = 'professional'
    domain_context: List[str] = field(default_factory=list)
    role_type: str = 'general'
    
    # Assessment requirements
    assessment_types_needed: List[str] = field(default_factory=list)
    balance_required: bool = False
    balance_confidence: float = 0.0
    
    # Processing metadata
    query_complexity: str = 'simple'
    processing_confidence: float = 0.0
    detected_sections: Dict[str, str] = field(default_factory=dict)
    
    # Priority scoring
    technical_priority: float = 0.5
    behavioral_priority: float = 0.5

class EnhancedQueryProcessor:
    """Advanced query processor with dynamic domain learning and sophisticated JD analysis"""
    
    def __init__(self, assessment_data: Optional[List[Dict]] = None):
        """Initialize with optional assessment data for dynamic learning"""
        
        # Load spaCy model for better NLP (fallback to basic if not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Dynamic domain learning from assessment data
        self.assessment_data = assessment_data or []
        self.learned_domains = {}
        self.learned_skills = {}
        
        # Enhanced pattern libraries
        self.section_patterns = self._build_section_patterns()
        self.priority_patterns = self._build_priority_patterns()
        self.balance_patterns = self._build_balance_patterns()
        
        # Initialize domain learning
        if self.assessment_data:
            self._learn_from_assessment_data()
        else:
            self._initialize_default_domains()
    
    def _learn_from_assessment_data(self):
        """Learn domains and skills from actual SHL assessment data"""
        logger.info("🧠 Learning domains from assessment data...")
        
        # Extract domains from assessment names and descriptions
        domain_keywords = {}
        skill_keywords = {}
        
        for assessment in self.assessment_data:
            name = assessment.get('name', '').lower()
            description = assessment.get('description', '').lower()
            test_types = assessment.get('test_type', [])
            
            # Extract potential domain keywords
            words = re.findall(r'\b[a-z]{3,}\b', name + ' ' + description)
            
            # Categorize by test type
            for test_type in test_types:
                if test_type not in domain_keywords:
                    domain_keywords[test_type] = Counter()
                
                for word in words:
                    if len(word) > 3 and word not in ['test', 'assessment', 'measures']:
                        domain_keywords[test_type][word] += 1
        
        # Build learned domains based on frequency
        self.learned_domains = {}
        for test_type, word_counts in domain_keywords.items():
            # Get top keywords for each test type
            top_words = [word for word, count in word_counts.most_common(20) if count > 1]
            if top_words:
                clean_test_type = test_type.lower().replace(' ', '_').replace('&', 'and')
                self.learned_domains[clean_test_type] = top_words
        
        # Enhanced domain mapping based on actual data
        self._build_enhanced_domains()
        
        logger.info(f"✅ Learned {len(self.learned_domains)} domain categories from assessment data")
    
    def _build_enhanced_domains(self):
        """Build comprehensive domain mappings from learned data"""
        
        # Technical domains - extracted from actual assessments
        self.technical_domains = {
            'programming': ['java', 'python', 'javascript', 'c++', 'c#', 'programming', 'coding', 'development'],
            'web_development': ['html', 'css', 'react', 'angular', 'web', 'frontend', 'backend', 'node'],
            'database': ['sql', 'mysql', 'postgresql', 'oracle', 'database', 'mongodb'],
            'cloud_devops': ['aws', 'docker', 'kubernetes', 'jenkins', 'cloud', 'devops'],
            'data_science': ['data', 'analytics', 'statistics', 'machine', 'learning', 'tableau'],
            'engineering': ['mechanical', 'civil', 'electrical', 'aerospace', 'chemical', 'engineering'],
            'healthcare': ['nursing', 'medical', 'pharmacology', 'healthcare', 'clinical', 'patient'],
            'finance': ['accounting', 'financial', 'banking', 'finance', 'audit', 'investment'],
            'science': ['chemistry', 'biology', 'physics', 'biochemistry', 'molecular'],
            'business': ['marketing', 'operations', 'management', 'business', 'strategy'],
            'design': ['photoshop', 'design', 'creative', 'graphics', 'visual'],
            'testing': ['qa', 'testing', 'selenium', 'automation', 'quality']
        }
        
        # Update with learned domains
        for domain_type, keywords in self.learned_domains.items():
            if 'knowledge' in domain_type or 'skills' in domain_type:
                # Extract specific technical domains
                if any(tech in keywords for tech in ['java', 'python', 'programming']):
                    self.technical_domains['programming'].extend(keywords[:10])
                elif any(web in keywords for web in ['html', 'css', 'web']):
                    self.technical_domains['web_development'].extend(keywords[:10])
                # Add more domain-specific mappings as needed
        
        # Behavioral domains
        self.behavioral_domains = {
            'communication': ['communication', 'communicate', 'verbal', 'written', 'presentation'],
            'collaboration': ['collaborate', 'collaboration', 'teamwork', 'team', 'cross-functional'],
            'leadership': ['leadership', 'lead', 'management', 'supervise', 'mentor', 'coaching'],
            'interpersonal': ['interpersonal', 'social', 'relationship', 'networking'],
            'customer_focus': ['customer', 'client', 'service', 'satisfaction'],
            'problem_solving': ['problem', 'solving', 'analytical', 'critical', 'thinking'],
            'adaptability': ['adaptable', 'flexible', 'change', 'agile'],
            'emotional_intelligence': ['emotional', 'empathy', 'self-awareness']
        }
        
        # Role contexts
        self.role_contexts = {
            'technical': ['developer', 'engineer', 'programmer', 'analyst', 'architect'],
            'business': ['manager', 'executive', 'director', 'consultant', 'strategist'],
            'administrative': ['admin', 'assistant', 'coordinator', 'clerk'],
            'financial': ['accountant', 'financial', 'finance', 'banking'],
            'healthcare': ['nurse', 'doctor', 'medical', 'clinical'],
            'creative': ['designer', 'creative', 'artist', 'marketing'],
            'sales': ['sales', 'business', 'development', 'account']
        }
    
    def _initialize_default_domains(self):
        """Initialize with default domains if no assessment data provided"""
        logger.info("📝 Initializing with default domain mappings")
        self._build_enhanced_domains()
    
    def _build_section_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for identifying JD sections"""
        return {
            'responsibilities': [
                r'responsibilities?', r'duties', r'role', r'you will', r'day[- ]to[- ]day',
                r'primary functions?', r'key activities', r'main tasks?'
            ],
            'requirements': [
                r'requirements?', r'qualifications?', r'must have', r'required',
                r'essential', r'mandatory', r'minimum', r'criteria'
            ],
            'preferred': [
                r'preferred', r'desired', r'nice[- ]?to[- ]?have', r'plus', r'bonus',
                r'additional', r'would be great', r'ideally'
            ],
            'experience': [
                r'experience', r'years?', r'background', r'track record',
                r'proven', r'demonstrated'
            ]
        }
    
    def _build_priority_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for determining skill priority"""
        return {
            'required': [
                r'must have', r'required', r'essential', r'mandatory',
                r'critical', r'necessary', r'needed'
            ],
            'preferred': [
                r'preferred', r'desired', r'ideal', r'would be great',
                r'nice to have', r'plus', r'bonus'
            ],
            'nice_to_have': [
                r'nice[- ]?to[- ]?have', r'additional', r'extra',
                r'would be a plus', r'bonus points'
            ]
        }
    
    def _build_balance_patterns(self) -> List[Dict[str, any]]:
        """Build advanced patterns for balance detection"""
        return [
            {
                'pattern': r'(?:technical|programming|development).{0,50}(?:and|with|plus).{0,50}(?:communication|collaboration|team)',
                'confidence': 0.9,
                'type': 'explicit_both'
            },
            {
                'pattern': r'(?:developer|engineer|programmer).{0,50}(?:who can|able to|with).{0,50}(?:collaborate|communicate|work with)',
                'confidence': 0.85,
                'type': 'role_plus_soft'
            },
            {
                'pattern': r'both.{0,30}(?:technical|hard).{0,30}(?:and|&).{0,30}(?:soft|interpersonal|communication)',
                'confidence': 0.95,
                'type': 'explicit_both_types'
            },
            {
                'pattern': r'(?:leadership|management).{0,50}(?:technical|programming|development)',
                'confidence': 0.8,
                'type': 'leadership_technical'
            },
            {
                'pattern': r'cross[- ]?functional.{0,30}(?:team|collaboration)',
                'confidence': 0.7,
                'type': 'cross_functional'
            }
        ]
    
    def process_query(self, input_text: str) -> Dict:
        """Enhanced query processing with sophisticated analysis"""
        
        logger.info(f"🔍 Processing query: '{input_text[:100]}...'")
        
        # Basic analysis
        query_type = self._detect_query_type(input_text)
        complexity = self._analyze_complexity(input_text)
        processed_content = self._extract_content(input_text, query_type)
        
        # Advanced analysis
        sections = self._analyze_sections(processed_content)
        skills_analysis = self._extract_skills_with_context(processed_content, sections)
        intent = self._build_enhanced_intent(skills_analysis, complexity)
        
        # Create intelligent search strategy
        search_strategy = self._create_advanced_search_strategy(intent)
        
        return {
            'original_query': input_text,
            'query_type': query_type,
            'complexity': complexity,
            'processed_content': processed_content,
            'sections': sections,
            'intent': intent,
            'search_strategy': search_strategy,
            'processing_metadata': {
                'skills_found': len(intent.technical_skills) + len(intent.behavioral_skills),
                'confidence': intent.processing_confidence,
                'balance_detected': intent.balance_required
            }
        }
    
    def _detect_query_type(self, input_text: str) -> str:
        """Enhanced query type detection"""
        if re.match(r'https?://', input_text.strip()):
            return 'url'
        
        word_count = len(input_text.split())
        
        # JD indicators
        jd_keywords = [
            'responsibilities', 'requirements', 'qualifications', 'job description',
            'role', 'position', 'we are looking for', 'candidate will',
            'years of experience', 'bachelor', 'master', 'degree'
        ]
        
        has_jd_keywords = any(keyword in input_text.lower() for keyword in jd_keywords)
        
        if word_count > 50 or has_jd_keywords:
            return 'job_description'
        elif word_count > 20:
            return 'detailed_query'
        else:
            return 'simple_query'
    
    def _analyze_complexity(self, input_text: str) -> str:
        """Enhanced complexity analysis"""
        word_count = len(input_text.split())
        sentence_count = len(sent_tokenize(input_text))
        
        # Check for complex structures
        has_lists = bool(re.search(r'[•\-\*]\s+|\d+\.\s+', input_text))
        has_sections = len(re.findall(r'\n\s*[A-Z]', input_text)) > 2
        
        if word_count > 200 or sentence_count > 10 or has_sections:
            return 'complex'
        elif word_count > 50 or sentence_count > 3 or has_lists:
            return 'medium'
        else:
            return 'simple'
    
    def _extract_content(self, input_text: str, query_type: str) -> str:
        """Enhanced content extraction with better cleaning"""
        if query_type == 'url':
            return self._scrape_job_posting(input_text)
        else:
            # Clean and normalize text
            text = re.sub(r'\s+', ' ', input_text)
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
            return text.strip()
    
    def _scrape_job_posting(self, url: str) -> str:
        """Enhanced URL scraping with better content extraction"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Try to find job description specific content
            job_content = soup.find('div', class_=re.compile(r'job|description|content', re.I))
            if job_content:
                text = job_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean and normalize
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 2)
            
            return text[:5000] if len(text) > 5000 else text
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape URL: {e}")
            return f"Failed to extract content from URL: {url}"
    
    def _analyze_sections(self, content: str) -> Dict[str, str]:
        """Analyze and identify different sections in job description"""
        
        sections = {}
        content_lower = content.lower()
        
        # Split content into potential sections
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        current_section = 'general'
        section_content = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Check if this paragraph starts a new section
            new_section = None
            for section_type, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, paragraph_lower):
                        new_section = section_type
                        break
                if new_section:
                    break
            
            if new_section:
                # Save previous section
                if section_content:
                    sections[current_section] = ' '.join(section_content)
                
                # Start new section
                current_section = new_section
                section_content = [paragraph]
            else:
                section_content.append(paragraph)
        
        # Save final section
        if section_content:
            sections[current_section] = ' '.join(section_content)
        
        return sections
    
    def _extract_skills_with_context(self, content: str, sections: Dict[str, str]) -> Dict[str, List[SkillMention]]:
        """Extract skills with detailed context analysis"""
        
        technical_skills = []
        behavioral_skills = []
        
        # Process each section
        for section_name, section_content in sections.items():
            
            # Extract technical skills
            for domain, keywords in self.technical_domains.items():
                for keyword in keywords:
                    if self._find_skill_in_context(keyword, section_content):
                        priority = self._determine_priority(keyword, section_content, section_name)
                        confidence = self._calculate_confidence(keyword, section_content)
                        
                        skill = SkillMention(
                            skill=keyword,
                            context=self._extract_context(keyword, section_content),
                            priority=priority,
                            section=section_name,
                            confidence=confidence
                        )
                        technical_skills.append(skill)
            
            # Extract behavioral skills
            for domain, keywords in self.behavioral_domains.items():
                for keyword in keywords:
                    if self._find_skill_in_context(keyword, section_content):
                        priority = self._determine_priority(keyword, section_content, section_name)
                        confidence = self._calculate_confidence(keyword, section_content)
                        
                        skill = SkillMention(
                            skill=keyword,
                            context=self._extract_context(keyword, section_content),
                            priority=priority,
                            section=section_name,
                            confidence=confidence
                        )
                        behavioral_skills.append(skill)
        
        return {
            'technical': technical_skills,
            'behavioral': behavioral_skills
        }
    
    def _find_skill_in_context(self, skill: str, content: str) -> bool:
        """Check if skill exists in content with proper word boundaries"""
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        return bool(re.search(pattern, content.lower()))
    
    def _determine_priority(self, skill: str, content: str, section: str) -> str:
        """Determine skill priority based on context and section"""
        
        content_lower = content.lower()
        
        # Check priority patterns around the skill
        skill_context = self._extract_context(skill, content, window=50)
        
        for priority, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if re.search(pattern, skill_context.lower()):
                    return priority
        
        # Default priority based on section
        if section in ['requirements', 'qualifications']:
            return 'required'
        elif section in ['preferred', 'nice_to_have']:
            return 'preferred'
        else:
            return 'required'  # Default assumption
    
    def _extract_context(self, skill: str, content: str, window: int = 100) -> str:
        """Extract context around a skill mention"""
        
        skill_lower = skill.lower()
        content_lower = content.lower()
        
        # Find skill position
        pos = content_lower.find(skill_lower)
        if pos == -1:
            return ""
        
        # Extract window around skill
        start = max(0, pos - window)
        end = min(len(content), pos + len(skill) + window)
        
        return content[start:end].strip()
    
    def _calculate_confidence(self, skill: str, content: str) -> float:
        """Calculate confidence score for skill detection"""
        
        # Base confidence
        confidence = 0.7
        
        # Boost for exact matches
        if skill.lower() in content.lower():
            confidence += 0.2
        
        # Boost for context indicators
        context = self._extract_context(skill, content)
        if any(word in context.lower() for word in ['experience', 'proficiency', 'knowledge', 'skills']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _build_enhanced_intent(self, skills_analysis: Dict, complexity: str) -> QueryIntent:
        """Build enhanced query intent with sophisticated analysis"""
        
        technical_skills = skills_analysis['technical']
        behavioral_skills = skills_analysis['behavioral']
        
        # Determine domain context
        domain_context = self._determine_domains(technical_skills)
        
        # Determine role type
        role_type = self._determine_role_type(technical_skills, behavioral_skills)
        
        # Determine experience level
        experience_level = self._determine_experience_level(technical_skills + behavioral_skills)
        
        # Advanced balance detection
        balance_result = self._detect_advanced_balance(technical_skills, behavioral_skills)
        
        # Calculate priority scores
        technical_priority, behavioral_priority = self._calculate_priority_scores(
            technical_skills, behavioral_skills, balance_result['required']
        )
        
        # Determine assessment types needed
        assessment_types = []
        if technical_skills or technical_priority > 0.3:
            assessment_types.append('Knowledge & Skills')
        if behavioral_skills or behavioral_priority > 0.3:
            assessment_types.append('Personality & Behavior')
        if not assessment_types:
            assessment_types = ['Knowledge & Skills', 'Personality & Behavior']
        
        return QueryIntent(
            technical_skills=technical_skills,
            behavioral_skills=behavioral_skills,
            experience_level=experience_level,
            domain_context=domain_context,
            role_type=role_type,
            assessment_types_needed=assessment_types,
            balance_required=balance_result['required'],
            balance_confidence=balance_result['confidence'],
            query_complexity=complexity,
            processing_confidence=self._calculate_overall_confidence(technical_skills, behavioral_skills),
            technical_priority=technical_priority,
            behavioral_priority=behavioral_priority
        )
    
    def _determine_domains(self, technical_skills: List[SkillMention]) -> List[str]:
        """Determine domain context from technical skills"""
        
        domain_scores = Counter()
        
        for skill in technical_skills:
            for domain, keywords in self.technical_domains.items():
                if skill.skill.lower() in [k.lower() for k in keywords]:
                    domain_scores[domain] += skill.confidence
        
        # Return top domains
        return [domain for domain, score in domain_scores.most_common(3)]
    
    def _determine_role_type(self, technical_skills: List[SkillMention], behavioral_skills: List[SkillMention]) -> str:
        """Determine role type based on skill mix"""
        
        if len(technical_skills) > len(behavioral_skills) * 2:
            return 'technical'
        elif len(behavioral_skills) > len(technical_skills) * 2:
            return 'behavioral'
        elif any('leadership' in skill.skill.lower() for skill in behavioral_skills):
            return 'leadership'
        else:
            return 'balanced'
    
    def _determine_experience_level(self, all_skills: List[SkillMention]) -> str:
        """Determine experience level from skill context"""
        
        level_indicators = {
            'entry': ['entry', 'junior', 'graduate', 'fresh', '0-2 years', 'beginner'],
            'senior': ['senior', 'lead', 'expert', '5+ years', 'advanced', 'principal'],
            'executive': ['manager', 'director', 'executive', 'vp', 'ceo', 'cto', 'coo']
        }
        
        for skill in all_skills:
            context_lower = skill.context.lower()
            for level, indicators in level_indicators.items():
                if any(indicator in context_lower for indicator in indicators):
                    return level
        
        return 'professional'  # Default
    
    def _detect_advanced_balance(self, technical_skills: List[SkillMention], behavioral_skills: List[SkillMention]) -> Dict[str, any]:
        """Advanced balance detection with confidence scoring"""
        
        # Check if both types of skills are present
        has_technical = len(technical_skills) > 0
        has_behavioral = len(behavioral_skills) > 0
        
        if not (has_technical and has_behavioral):
            return {'required': False, 'confidence': 0.0, 'reason': 'insufficient_skills'}
        
        # Check explicit balance patterns
        all_contexts = ' '.join([skill.context for skill in technical_skills + behavioral_skills])
        
        max_confidence = 0.0
        detected_pattern = None
        
        for pattern_info in self.balance_patterns:
            if re.search(pattern_info['pattern'], all_contexts, re.IGNORECASE):
                if pattern_info['confidence'] > max_confidence:
                    max_confidence = pattern_info['confidence']
                    detected_pattern = pattern_info['type']
        
        # Consider skill priorities
        high_priority_technical = sum(1 for s in technical_skills if s.priority == 'required')
        high_priority_behavioral = sum(1 for s in behavioral_skills if s.priority == 'required')
        
        if high_priority_technical > 0 and high_priority_behavioral > 0:
            max_confidence = max(max_confidence, 0.8)
            if not detected_pattern:
                detected_pattern = 'both_required'
        
        # Balance required if confidence is high enough
        balance_required = max_confidence > 0.6
        
        return {
            'required': balance_required,
            'confidence': max_confidence,
            'reason': detected_pattern or 'skill_mix'
        }
    
    def _calculate_priority_scores(self, technical_skills: List[SkillMention], 
                                 behavioral_skills: List[SkillMention], 
                                 balance_required: bool) -> Tuple[float, float]:
        """Calculate priority scores for technical vs behavioral assessments"""
        
        if balance_required:
            return 0.5, 0.5
        
        # Weight by number and priority of skills
        tech_score = 0.0
        behavioral_score = 0.0
        
        priority_weights = {'required': 1.0, 'preferred': 0.7, 'nice_to_have': 0.3}
        
        for skill in technical_skills:
            tech_score += priority_weights.get(skill.priority, 0.5) * skill.confidence
        
        for skill in behavioral_skills:
            behavioral_score += priority_weights.get(skill.priority, 0.5) * skill.confidence
        
        # Normalize
        total_score = tech_score + behavioral_score
        if total_score > 0:
            tech_score /= total_score
            behavioral_score /= total_score
        else:
            tech_score, behavioral_score = 0.5, 0.5
        
        return tech_score, behavioral_score
    
    def _calculate_overall_confidence(self, technical_skills: List[SkillMention], 
                                    behavioral_skills: List[SkillMention]) -> float:
        """Calculate overall processing confidence"""
        
        if not technical_skills and not behavioral_skills:
            return 0.2
        
        all_skills = technical_skills + behavioral_skills
        avg_confidence = sum(skill.confidence for skill in all_skills) / len(all_skills)
        
        # Boost confidence if multiple skills detected
        skill_count_boost = min(len(all_skills) * 0.1, 0.3)
        
        return min(avg_confidence + skill_count_boost, 1.0)
    
    def _create_advanced_search_strategy(self, intent: QueryIntent) -> Dict:
        """Create sophisticated search strategy based on enhanced intent"""
        
        search_queries = []
        
        if intent.balance_required:
            # Balanced approach with priority weighting
            if intent.technical_skills:
                tech_query = self._build_technical_query(intent.technical_skills, intent.domain_context)
                search_queries.append({
                    'query': tech_query,
                    'weight': intent.technical_priority,
                    'target_type': 'Knowledge & Skills',
                    'min_results': max(3, int(intent.technical_priority * 10))
                })
            
            if intent.behavioral_skills:
                behavioral_query = self._build_behavioral_query(intent.behavioral_skills, intent.experience_level)
                search_queries.append({
                    'query': behavioral_query,
                    'weight': intent.behavioral_priority,
                    'target_type': 'Personality & Behavior',
                    'min_results': max(3, int(intent.behavioral_priority * 10))
                })
        else:
            # Single focus with enhanced targeting
            if intent.technical_priority > intent.behavioral_priority:
                tech_query = self._build_technical_query(intent.technical_skills, intent.domain_context)
                search_queries.append({
                    'query': tech_query,
                    'weight': 1.0,
                    'target_type': 'Knowledge & Skills',
                    'min_results': 8
                })
            else:
                behavioral_query = self._build_behavioral_query(intent.behavioral_skills, intent.experience_level)
                search_queries.append({
                    'query': behavioral_query,
                    'weight': 1.0,
                    'target_type': 'Personality & Behavior',
                    'min_results': 8
                })
        
        return {
            'queries': search_queries,
            'balance_strategy': 'balanced' if intent.balance_required else 'focused',
            'target_assessments': intent.assessment_types_needed,
            'complexity': intent.query_complexity,
            'confidence': intent.processing_confidence
        }
    
    def _build_technical_query(self, technical_skills: List[SkillMention], domains: List[str]) -> str:
        """Build optimized technical query from skills and domains"""
        
        # Extract high-priority skills
        required_skills = [s.skill for s in technical_skills if s.priority == 'required']
        preferred_skills = [s.skill for s in technical_skills if s.priority == 'preferred']
        
        # Build query components
        query_parts = []
        
        # Add required skills with high weight
        if required_skills:
            query_parts.extend(required_skills[:3])  # Top 3 required skills
        
        # Add preferred skills
        if preferred_skills:
            query_parts.extend(preferred_skills[:2])  # Top 2 preferred skills
        
        # Add domain context
        if domains:
            query_parts.extend(domains[:2])  # Top 2 domains
        
        return ' '.join(query_parts[:5])  # Limit to 5 terms for focused search
    
    def _build_behavioral_query(self, behavioral_skills: List[SkillMention], experience_level: str) -> str:
        """Build optimized behavioral query from skills and experience level"""
        
        # Extract skill categories
        skill_categories = []
        for skill in behavioral_skills:
            for category, keywords in self.behavioral_domains.items():
                if skill.skill.lower() in [k.lower() for k in keywords]:
                    skill_categories.append(category)
                    break
        
        # Build query components
        query_parts = []
        
        # Add unique skill categories
        unique_categories = list(set(skill_categories))
        query_parts.extend(unique_categories[:3])
        
        # Add experience level context
        if experience_level != 'professional':
            query_parts.append(experience_level)
        
        # Add behavioral assessment indicators
        query_parts.extend(['personality', 'behavior'])
        
        return ' '.join(query_parts[:5])
    
    def get_query_summary(self, processed_query: Dict) -> Dict:
        """Generate a human-readable summary of the processed query"""
        
        intent = processed_query['intent']
        
        summary = {
            'query_type': processed_query['query_type'],
            'complexity': processed_query['complexity'],
            'skills_detected': {
                'technical': len(intent.technical_skills),
                'behavioral': len(intent.behavioral_skills)
            },
            'domains': intent.domain_context,
            'experience_level': intent.experience_level,
            'role_type': intent.role_type,
            'balance_required': intent.balance_required,
            'balance_confidence': intent.balance_confidence,
            'assessment_strategy': processed_query['search_strategy']['balance_strategy'],
            'recommended_assessments': intent.assessment_types_needed,
            'processing_confidence': intent.processing_confidence
        }
        
        # Add skill details
        if intent.technical_skills:
            summary['top_technical_skills'] = [
                {
                    'skill': skill.skill,
                    'priority': skill.priority,
                    'confidence': skill.confidence
                }
                for skill in sorted(intent.technical_skills, key=lambda x: x.confidence, reverse=True)[:5]
            ]
        
        if intent.behavioral_skills:
            summary['top_behavioral_skills'] = [
                {
                    'skill': skill.skill,
                    'priority': skill.priority,
                    'confidence': skill.confidence
                }
                for skill in sorted(intent.behavioral_skills, key=lambda x: x.confidence, reverse=True)[:5]
            ]
        
        return summary

# Integration helper for recommender system
class QueryProcessorIntegration:
    """Helper class to integrate query processor with recommender system"""
    
    def __init__(self, query_processor: EnhancedQueryProcessor):
        self.query_processor = query_processor
    
    def process_for_recommendation(self, query: str, assessment_data: List[Dict]) -> Dict:
        """Process query and format for recommender system"""
        
        # Process the query
        processed = self.query_processor.process_query(query)
        intent = processed['intent']
        
        # Format for recommender
        recommendation_input = {
            'query': query,
            'processed_content': processed['processed_content'],
            'technical_skills': [skill.skill for skill in intent.technical_skills],
            'behavioral_skills': [skill.skill for skill in intent.behavioral_skills],
            'domains': intent.domain_context,
            'experience_level': intent.experience_level,
            'balance_required': intent.balance_required,
            'assessment_types_needed': intent.assessment_types_needed,
            'search_strategy': processed['search_strategy'],
            'priority_weights': {
                'technical': intent.technical_priority,
                'behavioral': intent.behavioral_priority
            }
        }
        
        return recommendation_input
    
    def enhance_recommendations(self, recommendations: List[Dict], processed_query: Dict) -> List[Dict]:
        """Enhance recommendations with query processing insights"""
        
        intent = processed_query['intent']
        
        enhanced_recommendations = []
        for rec in recommendations:
            enhanced_rec = rec.copy()
            
            # Add relevance scoring based on query analysis
            relevance_score = self._calculate_relevance_score(rec, intent)
            enhanced_rec['query_relevance'] = relevance_score
            
            # Add explanation
            explanation = self._generate_explanation(rec, intent)
            enhanced_rec['selection_reason'] = explanation
            
            enhanced_recommendations.append(enhanced_rec)
        
        return enhanced_recommendations
    
    def _calculate_relevance_score(self, recommendation: Dict, intent: QueryIntent) -> float:
        """Calculate how relevant a recommendation is to the query intent"""
        
        score = 0.0
        
        # Test type alignment
        rec_test_types = recommendation.get('test_type', [])
        if intent.balance_required:
            # For balanced queries, prefer assessments that match the needed types
            if 'Knowledge & Skills' in intent.assessment_types_needed and any('Knowledge' in t for t in rec_test_types):
                score += 0.4
            if 'Personality & Behavior' in intent.assessment_types_needed and any('Personality' in t or 'Behavior' in t for t in rec_test_types):
                score += 0.4
        else:
            # For focused queries, boost primary type
            if intent.technical_priority > intent.behavioral_priority:
                if any('Knowledge' in t for t in rec_test_types):
                    score += 0.6
            else:
                if any('Personality' in t or 'Behavior' in t for t in rec_test_types):
                    score += 0.6
        
        # Domain alignment
        rec_name = recommendation.get('name', '').lower()
        rec_desc = recommendation.get('description', '').lower()
        
        for domain in intent.domain_context:
            if domain in rec_name or domain in rec_desc:
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _generate_explanation(self, recommendation: Dict, intent: QueryIntent) -> str:
        """Generate explanation for why this assessment was selected"""
        
        reasons = []
        
        # Test type reasoning
        rec_test_types = recommendation.get('test_type', [])
        if intent.balance_required:
            if any('Knowledge' in t for t in rec_test_types) and intent.technical_skills:
                reasons.append("matches technical skill requirements")
            if any('Personality' in t or 'Behavior' in t for t in rec_test_types) and intent.behavioral_skills:
                reasons.append("assesses behavioral competencies")
        
        # Domain reasoning
        rec_name = recommendation.get('name', '').lower()
        for domain in intent.domain_context:
            if domain in rec_name:
                reasons.append(f"relevant to {domain} domain")
                break
        
        # Experience level reasoning
        if intent.experience_level in ['senior', 'executive'] and 'advanced' in rec_name:
            reasons.append("suitable for senior-level positions")
        elif intent.experience_level == 'entry' and any(word in rec_name for word in ['basic', 'foundation', 'entry']):
            reasons.append("appropriate for entry-level candidates")
        
        if not reasons:
            reasons = ["matches general assessment criteria"]
        
        return "; ".join(reasons)

# Example usage and testing
if __name__ == "__main__":
    # Example with assessment data
    sample_assessments = [
        {
            'name': 'Python Programming Assessment',
            'description': 'Comprehensive test measuring Python programming skills, database knowledge, and web development capabilities',
            'test_type': ['Knowledge & Skills']
        },
        {
            'name': 'Leadership & Communication Assessment',
            'description': 'Evaluates leadership potential, communication skills, and team collaboration abilities',
            'test_type': ['Personality & Behavior']
        }
    ]
    
    # Initialize processor
    processor = EnhancedQueryProcessor(assessment_data=sample_assessments)
    
    # Test queries
    test_queries = [
        "I need a Java developer who can collaborate effectively with business teams",
        "Looking for a senior Python developer with strong database skills",
        "https://example.com/job-posting/senior-software-engineer",
        """
        We are seeking a Software Engineer with the following requirements:
        - 3+ years of Python programming experience
        - Strong SQL and database design skills
        - Experience with REST APIs and web development
        - Excellent communication and teamwork abilities
        - Bachelor's degree in Computer Science or related field
        
        Preferred qualifications:
        - Experience with cloud platforms (AWS, Azure)
        - Knowledge of machine learning frameworks
        - Leadership experience
        """
    ]
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST QUERY {i}")
        print(f"{'='*60}")
        
        try:
            result = processor.process_query(query)
            summary = processor.get_query_summary(result)
            
            print(f"Query Type: {summary['query_type']}")
            print(f"Complexity: {summary['complexity']}")
            print(f"Skills Detected: {summary['skills_detected']}")
            print(f"Domains: {summary['domains']}")
            print(f"Role Type: {summary['role_type']}")
            print(f"Balance Required: {summary['balance_required']} (confidence: {summary['balance_confidence']:.2f})")
            print(f"Assessment Strategy: {summary['assessment_strategy']}")
            print(f"Recommended Types: {summary['recommended_assessments']}")
            print(f"Processing Confidence: {summary['processing_confidence']:.2f}")
            
            if 'top_technical_skills' in summary:
                print(f"\nTop Technical Skills:")
                for skill in summary['top_technical_skills']:
                    print(f"  - {skill['skill']} ({skill['priority']}, {skill['confidence']:.2f})")
            
            if 'top_behavioral_skills' in summary:
                print(f"\nTop Behavioral Skills:")
                for skill in summary['top_behavioral_skills']:
                    print(f"  - {skill['skill']} ({skill['priority']}, {skill['confidence']:.2f})")
                    
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")