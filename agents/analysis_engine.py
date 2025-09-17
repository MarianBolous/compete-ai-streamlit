"""
AI Analysis Engine - Enhanced Agent 2 for Competitive Intelligence Analysis

This module implements a comprehensive AI-powered analysis engine that transforms raw
competitive intelligence data into strategic insights and recommendations.

Features:
- Timeout handling for all operations
- Loop & evaluation policy with configurable retries
- Partial results on failure with graceful degradation  
- Delta tracking for historical comparisons
- Standardized callback interfaces
- Robust result schemas with dataclasses
- Structured logging with contexts
- Production-ready error handling

Usage:
    from agents.analysis_engine import AIAnalysisEngine, AnalysisConfig, AnalysisResult
    
    engine = AIAnalysisEngine(config=AnalysisConfig())
    result = engine.analyze_competitor(company_name, company_data, user_context)
"""

from __future__ import annotations

# ============================================================
# IMPORTS
# ============================================================

import asyncio
import logging
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Protocol, Union, Tuple
from contextlib import contextmanager
import traceback

# Optional dependencies - graceful fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# Set up structured logging
logger = logging.getLogger(__name__)

# Env + SDK clients
import os
from dotenv import load_dotenv
from openai import OpenAI

# LOAD ENV ONCE
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# INTERFACES AND PROTOCOLS
# ============================================================

class UICallbacks(Protocol):
    """Standardized callback interface for UI progress updates."""
    def on_progress(self, pct: int, message: str) -> None:
        """Called during analysis progress."""
        ...
    
    def on_done(self) -> None:
        """Called when analysis is complete."""
        ...

class AnalysisCallbacks(Protocol):
    """Callbacks for analysis-specific events."""
    def on_stage_start(self, stage: str, details: Dict[str, Any]) -> None:
        """Called when analysis stage starts."""
        ...
        
    def on_stage_complete(self, stage: str, result: Dict[str, Any], duration: float) -> None:
        """Called when analysis stage completes."""
        ...
        
    def on_error(self, stage: str, error: Exception, partial_result: Optional[Dict[str, Any]] = None) -> None:
        """Called when an error occurs."""
        ...

# ============================================================
# CONFIGURATION AND SCHEMAS
# ============================================================

@dataclass
class AnalysisConfig:
    """Configuration for AI Analysis Engine."""
    # Timeout settings
    default_timeout_sec: float = 30.0
    analysis_timeout_sec: float = 45.0
    tool_timeout_sec: float = 15.0
    
    # Loop and evaluation policy
    max_plan_loops: int = 3
    confidence_threshold: float = 0.7
    retry_on_low_confidence: bool = True
    
    # Feature flags
    enable_embeddings: bool = True
    enable_financial_analysis: bool = True
    enable_swot_analysis: bool = True
    enable_competitive_analysis: bool = True
    enable_delta_tracking: bool = True
    
    # Tool configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_embeddings: bool = True
    max_cache_size: int = 1000
    
    # Fallback settings
    graceful_degradation: bool = True
    return_partial_on_failure: bool = True
    min_required_sections: List[str] = field(default_factory=lambda: ["market_positioning", "swot_analysis"])

    # ---------------- NEW: LLM refinement & routing config ----------------
    enable_llm_refinement: bool = False  # turn on to enable multi-model JSON refinement
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-20240620"
    # Map stage -> provider ("openai" | "anthropic" | "auto")
    llm_provider_per_stage: Dict[str, str] = field(default_factory=lambda: {
        "swot_analysis": "auto",
        "market_positioning": "openai",
        "competitive_landscape": "anthropic",
        "strategic_recommendations": "auto",
        "threat_assessment": "auto",
    })
    # ----------------------------------------------------------------------

@dataclass
class AnalysisMetadata:
    """Metadata about the analysis process."""
    analysis_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_sec: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0
    loops_executed: int = 0

@dataclass
class AnalysisResult:
    """Standardized analysis result schema."""
    # Core analysis sections
    competitor_name: str
    market_positioning: Dict[str, Any] = field(default_factory=dict)
    swot_analysis: Dict[str, Any] = field(default_factory=dict)
    competitive_landscape: Dict[str, Any] = field(default_factory=dict)
    strategic_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    threat_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Quality and confidence metrics
    confidence_score: float = 0.0
    section_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Analysis metadata
    metadata: AnalysisMetadata = field(default_factory=lambda: AnalysisMetadata(
        analysis_id="", 
        started_at=datetime.now()
    ))
    
    # Status tracking
    status: str = "completed"  # completed, partial, failed
    analysis_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Delta tracking
    changes_from_previous: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def is_complete(self) -> bool:
        """Check if analysis is complete and meets minimum requirements."""
        return (
            self.status == "completed" and 
            self.confidence_score >= 0.5 and
            bool(self.market_positioning) and
            bool(self.swot_analysis)
        )

# ============================================================
# UTILITY FUNCTIONS AND HELPERS
# ============================================================

def safe_get(data: Dict, *keys, default=None):
    """Safely navigate nested dictionaries."""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def calculate_confidence(scores: List[float]) -> float:
    """Calculate weighted confidence from multiple scores."""
    if not scores:
        return 0.0
    valid_scores = [s for s in scores if 0 <= s <= 1]
    if not valid_scores:
        return 0.0
    return round(sum(valid_scores) / len(valid_scores), 3)

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(str(text or "").strip().split()).lower()

def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

@contextmanager
def logging_context(stage: str, **kwargs):
    """Context manager for structured logging."""
    start_time = time.time()
    logger.info(f"Starting {stage}", extra={"stage": stage, **kwargs})
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {stage}", extra={"stage": stage, "duration": duration, **kwargs})
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed {stage}", extra={"stage": stage, "duration": duration, "error": str(e), **kwargs})
        raise

# ============================================================
# ANALYSIS TOOLS
# ============================================================

class EmbeddingTool:
    """Handles text embeddings with caching and fallbacks."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.model = None
        self.cache = {}
        
        if config.enable_embeddings and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(config.embedding_model)
                logger.info(f"Loaded embedding model: {config.embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.model = None
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """Get similarity between two texts."""
        if not self.model:
            # Fallback to simple text overlap
            words1 = set(normalize_text(text1).split())
            words2 = set(normalize_text(text2).split())
            return calculate_jaccard_similarity(words1, words2)
        
        try:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            
            if emb1 is None or emb2 is None:
                return 0.0
                
            # Cosine similarity
            dot_product = np.dot(emb1, emb2) if HAS_NUMPY else sum(a * b for a, b in zip(emb1, emb2))
            norm1 = np.linalg.norm(emb1) if HAS_NUMPY else (sum(x**2 for x in emb1) ** 0.5)
            norm2 = np.linalg.norm(emb2) if HAS_NUMPY else (sum(x**2 for x in emb2) ** 0.5)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product) / float(norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")
            return 0.0
    
    def _get_embedding(self, text: str) -> Optional[Any]:
        """Get embedding for text with caching."""
        if not self.model:
            return None
            
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest() if self.config.cache_embeddings else None
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            
            # Cache if enabled
            if cache_key:
                if len(self.cache) >= self.config.max_cache_size:
                    # Remove oldest entry
                    self.cache.pop(next(iter(self.cache)))
                self.cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

class SWOTAnalyzer:
    """SWOT analysis with configurable depth and AI enhancement."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def analyze(self, company_name: str, company_data: Dict, user_context: Dict) -> Dict[str, Any]:
        """Generate SWOT analysis."""
        try:
            with logging_context("swot_analysis", company=company_name):
                # Extract key data points
                product_info = safe_get(company_data, "product_info", default={})
                financial_data = safe_get(company_data, "financial_data", default={})
                news = safe_get(company_data, "recent_news", default=[])
                reviews = safe_get(company_data, "reviews", default={})
                
                # Generate SWOT components
                strengths = self._identify_strengths(product_info, financial_data, reviews, news)
                weaknesses = self._identify_weaknesses(product_info, financial_data, reviews)
                opportunities = self._identify_opportunities(user_context, product_info, news)
                threats = self._identify_threats(user_context, financial_data, news)
                
                return {
                    "strengths": strengths[:5],  # Limit to top 5
                    "weaknesses": weaknesses[:5],
                    "opportunities": opportunities[:5],
                    "threats": threats[:5],
                    "analysis_confidence": self._calculate_swot_confidence(
                        strengths, weaknesses, opportunities, threats
                    )
                }
                
        except Exception as e:
            logger.error(f"SWOT analysis failed: {e}")
            return self._fallback_swot(company_name)
    
    def _identify_strengths(self, product_info: Dict, financial_data: Dict, 
                          reviews: Dict, news: List) -> List[str]:
        """Identify company strengths."""
        strengths = []
        
        # Product strengths
        features = safe_get(product_info, "key_features", default=[])
        if len(features) > 5:
            strengths.append("Comprehensive feature set with strong product capabilities")
        
        if any("ai" in normalize_text(f) for f in features):
            strengths.append("Advanced AI and machine learning capabilities")
        
        # Financial strengths
        if safe_get(financial_data, "total_funding"):
            strengths.append("Strong financial backing enabling growth and innovation")
        
        # Customer satisfaction
        avg_rating = safe_get(reviews, "avg_rating", default=0)
        if avg_rating >= 4.0:
            strengths.append(f"High customer satisfaction with {avg_rating:.1f}/5.0 average rating")
        
        # Market momentum
        if len(news) >= 3:
            strengths.append("Strong market presence with consistent media coverage")
        
        # Default strengths if none identified
        if not strengths:
            strengths = [
                "Established market presence and brand recognition",
                "Focused product development and clear value proposition"
            ]
        
        return strengths
    
    def _identify_weaknesses(self, product_info: Dict, financial_data: Dict, 
                           reviews: Dict) -> List[str]:
        """Identify potential weaknesses."""
        weaknesses = []
        
        # Limited feature set
        features = safe_get(product_info, "key_features", default=[])
        if len(features) < 3:
            weaknesses.append("Limited feature set compared to full-service competitors")
        
        # Pricing concerns
        if "freemium" in normalize_text(safe_get(product_info, "pricing_model", "")):
            weaknesses.append("Freemium model may limit revenue potential and customer commitment")
        
        # Customer feedback
        avg_rating = safe_get(reviews, "avg_rating", default=0)
        if avg_rating > 0 and avg_rating < 3.5:
            weaknesses.append(f"Below-average customer satisfaction at {avg_rating:.1f}/5.0")
        
        # Financial constraints
        if not safe_get(financial_data, "total_funding"):
            weaknesses.append("Limited funding may constrain growth and competitive response")
        
        # Default weaknesses
        if not weaknesses:
            weaknesses = [
                "Market position still developing against established competitors",
                "Resource constraints typical of growing companies"
            ]
        
        return weaknesses
    
    def _identify_opportunities(self, user_context: Dict, product_info: Dict, 
                              news: List) -> List[str]:
        """Identify market opportunities."""
        opportunities = [
            "Enterprise market expansion with specialized features and support",
            "International expansion to underserved geographic markets",
            "Strategic partnerships to expand distribution and capabilities",
            "Vertical market specialization for industry-specific solutions",
            "AI and automation trends creating demand for intelligent solutions"
        ]
        
        # Context-specific opportunities
        industry = safe_get(user_context, "industry", "")
        if industry in ["FinTech", "HealthTech"]:
            opportunities.insert(0, f"Regulatory compliance features for {industry} market")
        
        return opportunities
    
    def _identify_threats(self, user_context: Dict, financial_data: Dict, 
                         news: List) -> List[str]:
        """Identify potential threats."""
        threats = [
            "Well-funded competitors with deeper resources and market presence",
            "Economic downturn reducing B2B software spending and lengthening sales cycles",
            "Big Tech platforms (Microsoft, Google) offering competing solutions at scale",
            "Regulatory changes affecting data privacy, AI governance, and compliance requirements",
            "Market saturation leading to increased price competition and customer acquisition costs"
        ]
        
        # Industry-specific threats
        industry = safe_get(user_context, "industry", "")
        if industry == "AI/ML":
            threats.insert(0, "Rapidly evolving AI technology making current solutions obsolete")
        
        return threats
    
    def _calculate_swot_confidence(self, strengths: List, weaknesses: List, 
                                 opportunities: List, threats: List) -> float:
        """Calculate confidence in SWOT analysis."""
        # Base confidence on data availability
        total_items = len(strengths) + len(weaknesses) + len(opportunities) + len(threats)
        if total_items >= 15:
            return 0.9
        elif total_items >= 10:
            return 0.8
        elif total_items >= 8:
            return 0.7
        else:
            return 0.6
    
    def _fallback_swot(self, company_name: str) -> Dict[str, Any]:
        """Fallback SWOT when analysis fails."""
        return {
            "strengths": [f"{company_name} has established market presence"],
            "weaknesses": ["Limited data available for comprehensive analysis"],
            "opportunities": ["Market expansion and product development"],
            "threats": ["Competitive pressure and market changes"],
            "analysis_confidence": 0.5
        }

class CompetitiveAnalyzer:
    """Competitive landscape and positioning analysis."""
    
    def __init__(self, config: AnalysisConfig, embedding_tool: EmbeddingTool):
        self.config = config
        self.embedding_tool = embedding_tool
    
    def analyze_positioning(self, company_name: str, company_data: Dict, 
                          user_context: Dict) -> Dict[str, Any]:
        """Analyze market positioning."""
        try:
            with logging_context("market_positioning", company=company_name):
                product_info = safe_get(company_data, "product_info", default={})
                financial_data = safe_get(company_data, "financial_data", default={})
                
                # Determine market segment
                industry = safe_get(user_context, "industry", "Technology")
                segment = f"{industry} Software"
                
                # Assess position based on funding and features
                features = safe_get(product_info, "key_features", default=[])
                funding = safe_get(financial_data, "total_funding", "")
                
                position = self._determine_position(features, funding)
                market_share = self._estimate_market_share(funding)
                growth_trajectory = self._assess_growth(financial_data)
                advantages = self._identify_advantages(features, financial_data)
                
                return {
                    "segment": segment,
                    "position": position,
                    "market_share": market_share,
                    "growth_trajectory": growth_trajectory,
                    "competitive_advantages": advantages
                }
                
        except Exception as e:
            logger.error(f"Market positioning analysis failed: {e}")
            return self._fallback_positioning(company_name, user_context)
    
    def analyze_landscape(self, company_name: str, company_data: Dict, 
                         user_context: Dict) -> Dict[str, Any]:
        """Analyze competitive landscape."""
        try:
            with logging_context("competitive_landscape", company=company_name):
                # Generate competitor lists
                direct_competitors = self._identify_direct_competitors(user_context)
                indirect_competitors = self._identify_indirect_competitors(user_context)
                
                # Analyze competitive positioning
                product_info = safe_get(company_data, "product_info", default={})
                moats = self._identify_moats(product_info)
                vulnerabilities = self._identify_vulnerabilities(product_info, user_context)
                
                return {
                    "direct_competitors": direct_competitors,
                    "indirect_competitors": indirect_competitors,
                    "competitive_moats": moats,
                    "vulnerability_areas": vulnerabilities
                }
                
        except Exception as e:
            logger.error(f"Competitive landscape analysis failed: {e}")
            return self._fallback_landscape()
    
    def _determine_position(self, features: List, funding: str) -> str:
        """Determine market position based on capabilities."""
        feature_count = len(features)
        has_funding = bool(funding)
        
        if feature_count >= 8 and has_funding:
            return "Market leader with comprehensive solution and strong backing"
        elif feature_count >= 5 and has_funding:
            return "Growing challenger with strong feature set and investment"
        elif feature_count >= 5:
            return "Established player with solid product capabilities"
        elif has_funding:
            return "Emerging player with financial backing for growth"
        else:
            return "Developing competitor building market position"
    
    def _estimate_market_share(self, financial_data: Dict) -> str:
        """Estimate market share based on funding level."""
        funding = safe_get(financial_data, "total_funding", "")
        funding_str = str(funding or "")  # <-- robust against None

        if "100M" in funding_str or "B" in funding_str:
            return "10-15% in target segments"
        elif any(x in funding_str for x in ["50M", "75M"]):
            return "5-10% in target segments"
        elif funding_str:
            return "2-5% in target segments"
        else:
            return "Under 2% market share"
    
    def _assess_growth(self, financial_data: Dict) -> str:
        """Assess growth trajectory."""
        growth_rate = safe_get(financial_data, "growth_rate", "")
        
        if growth_rate:
            # Extract percentage if present
            import re
            numbers = re.findall(r'(\d+)', growth_rate)
            if numbers and int(numbers[0]) >= 100:
                return f"Rapid growth trajectory with {growth_rate}"
            elif numbers and int(numbers[0]) >= 50:
                return f"Strong growth with {growth_rate}"
            else:
                return f"Steady growth at {growth_rate}"
        
        return "Growth trajectory to be determined"
    
    def _identify_advantages(self, features: List, financial_data: Dict) -> List[str]:
        """Identify competitive advantages."""
        advantages = []
        
        # Technology advantages
        feature_text = " ".join([normalize_text(str(f)) for f in features])
        if "ai" in feature_text or "machine learning" in feature_text:
            advantages.append("Advanced AI and automation capabilities")
        
        if "api" in feature_text or "integration" in feature_text:
            advantages.append("Strong integration ecosystem and API platform")
        
        # Financial advantages
        if safe_get(financial_data, "total_funding"):
            advantages.append("Strong financial position enabling rapid development and market expansion")
        
        # Default advantages
        if not advantages:
            advantages = [
                "Focused product development and user experience",
                "Agile development and rapid feature deployment"
            ]
        
        return advantages[:4]  # Limit to 4
    
    def _identify_direct_competitors(self, user_context: Dict) -> List[str]:
        """Identify direct competitors by industry."""
        industry = safe_get(user_context, "industry", "SaaS")
        
        competitor_map = {
            "SaaS": ["Salesforce", "HubSpot", "Zendesk"],
            "E-commerce": ["Shopify", "BigCommerce", "WooCommerce"],
            "FinTech": ["Stripe", "Square", "Plaid"],
            "HealthTech": ["Epic", "Cerner", "Veracyte"],
            "EdTech": ["Canvas", "Blackboard", "Coursera"],
            "MarTech": ["Marketo", "Pardot", "Mailchimp"],
            "AI/ML": ["Palantir", "DataRobot", "H2O.ai"],
        }
        
        return competitor_map.get(industry, ["Competitor A", "Competitor B", "Competitor C"])
    
    def _identify_indirect_competitors(self, user_context: Dict) -> List[str]:
        """Identify indirect competitors."""
        return [
            "Microsoft Office 365 suite",
            "Google Workspace ecosystem",
            "Custom internal solutions"
        ]
    
    def _identify_moats(self, product_info: Dict) -> List[str]:
        """Identify competitive moats."""
        features = safe_get(product_info, "key_features", default=[])
        feature_text = " ".join([normalize_text(f) for f in features])
        
        moats = []
        
        if "ai" in feature_text:
            moats.append("Proprietary AI algorithms and machine learning models")
        
        if "data" in feature_text or "analytics" in feature_text:
            moats.append("Proprietary data insights and analytics capabilities")
        
        if "integration" in feature_text:
            moats.append("Extensive integration ecosystem creating switching costs")
        
        # Default moats
        if not moats:
            moats = [
                "Customer data and workflow integration creating switching costs",
                "Specialized domain expertise and feature depth"
            ]
        
        return moats
    
    def _identify_vulnerabilities(self, product_info: Dict, user_context: Dict) -> List[str]:
        """Identify vulnerability areas."""
        return [
            "Price competition from well-funded new entrants",
            "Feature commoditization by larger platform players",
            "Customer acquisition cost escalation in competitive market",
            "Talent retention in competitive hiring environment"
        ]
    
    def _fallback_positioning(self, company_name: str, user_context: Dict) -> Dict[str, Any]:
        """Fallback positioning when analysis fails."""
        industry = safe_get(user_context, "industry", "Technology")
        return {
            "segment": f"{industry} Software",
            "position": "Market participant with developing position",
            "market_share": "Position to be determined",
            "growth_trajectory": "Assessment pending",
            "competitive_advantages": ["Analysis in progress"]
        }
    
    def _fallback_landscape(self) -> Dict[str, Any]:
        """Fallback landscape when analysis fails."""
        return {
            "direct_competitors": ["Analysis pending"],
            "indirect_competitors": ["Analysis pending"],
            "competitive_moats": ["Assessment required"],
            "vulnerability_areas": ["Analysis in progress"]
        }

class ThreatAnalyzer:
    """Threat assessment and strategic risk analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def analyze_threats(self, company_name: str, company_data: Dict, 
                       analysis_results: Dict, user_context: Dict) -> Dict[str, Any]:
        """Comprehensive threat analysis."""
        try:
            with logging_context("threat_assessment", company=company_name):
                # Assess threat level
                threat_level = self._assess_overall_threat_level(company_data, analysis_results)
                
                # Identify key threat factors
                threat_factors = self._identify_threat_factors(company_data, analysis_results)
                
                # Generate mitigation strategies
                mitigation_strategies = self._generate_mitigation_strategies(threat_factors, user_context)
                
                return {
                    "overall_threat_level": threat_level,
                    "key_threat_factors": threat_factors,
                    "mitigation_strategies": mitigation_strategies,
                    "risk_matrix": self._generate_risk_matrix(threat_factors)
                }
                
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            return self._fallback_threat_assessment()
    
    def _assess_overall_threat_level(self, company_data: Dict, analysis_results: Dict) -> str:
        """Assess overall competitive threat level."""
        threat_score = 0
    
        # Financial threat
        financial_data = safe_get(company_data, "financial_data", default={})
        if safe_get(financial_data, "total_funding"):
            threat_score += 2
    
        # Product threat
        features = safe_get(company_data, "product_info", "key_features", default=[]) or []
        if len(features) >= 8:
            threat_score += 3
        elif len(features) >= 5:
            threat_score += 2
        elif len(features) >= 3:
            threat_score += 1
    
        # Market momentum
        news = safe_get(company_data, "recent_news", default=[]) or []
        if len(news) >= 5:
            threat_score += 2
        elif len(news) >= 3:
            threat_score += 1
    
        # Customer satisfaction (robust float coercion)
        reviews = safe_get(company_data, "reviews", default={}) or {}
        avg_rating = safe_get(reviews, "avg_rating", default=0)
        try:
            rating = float(avg_rating)
        except (TypeError, ValueError):
            rating = 0.0
    
        if rating >= 4.5:
            threat_score += 2
        elif rating >= 4.0:
            threat_score += 1
    
        # Convert score to threat level
        if threat_score >= 7:
            return "Critical"
        elif threat_score >= 5:
            return "High"
        elif threat_score >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _identify_threat_factors(self, company_data: Dict, analysis_results: Dict) -> List[str]:
        """Identify specific threat factors."""
        factors = []
        
        # Financial threat factors
        financial_data = safe_get(company_data, "financial_data", default={})
        if safe_get(financial_data, "total_funding"):
            factors.append("Strong financial position enabling aggressive market expansion and competitive responses")
        
        # Product threat factors
        features = safe_get(company_data, "product_info", "key_features", default=[])
        if len(features) >= 5:
            factors.append("Comprehensive product capabilities creating pressure on feature parity")
        
        # Market presence factors
        news = safe_get(company_data, "recent_news", default=[])
        if len(news) >= 3:
            factors.append("Active market presence and strong brand momentum affecting customer mindshare")
        
        # Customer loyalty factors
        reviews = safe_get(company_data, "reviews", default={})
        avg_rating = safe_get(reviews, "avg_rating", default=0)
        try:
            rating = float(avg_rating) if avg_rating is not None else 0.0
            if rating >= 4.0:
                factors.append("High customer satisfaction creating loyalty and reducing switching likelihood")
        except (TypeError, ValueError):
            pass  # Skip if rating cannot be converted to float
        
        # Default factors if none identified
        if not factors:
            factors = [
                "Competitive market positioning requiring strategic response",
                "Market presence creating awareness and consideration challenges"
            ]
        
        return factors[:4]  # Limit to top 4
    
    def _generate_mitigation_strategies(self, threat_factors: List[str], user_context: Dict) -> List[str]:
        """Generate mitigation strategies for identified threats."""
        strategies = [
            "Strengthen unique value proposition and competitive differentiation messaging",
            "Accelerate product development in areas of competitive advantage",
            "Enhance customer success and retention programs to reduce churn risk",
            "Build strategic partnerships to expand market reach and capabilities",
            "Monitor competitive moves and develop rapid response capabilities"
        ]
        
        # Add context-specific strategies
        company_stage = safe_get(user_context, "company_stage", "")
        if company_stage in ["Early Stage", "Growth Stage"]:
            strategies.insert(0, "Focus on niche market segments where competitive pressure is lower")
        
        return strategies[:5]  # Limit to 5
    
    def _generate_risk_matrix(self, threat_factors: List[str]) -> Dict[str, List[str]]:
        """Generate risk matrix categorization."""
        return {
            "high_impact_high_probability": threat_factors[:2] if threat_factors else [],
            "high_impact_low_probability": [],
            "low_impact_high_probability": threat_factors[2:] if len(threat_factors) > 2 else [],
            "low_impact_low_probability": []
        }
    
    def _fallback_threat_assessment(self) -> Dict[str, Any]:
        """Fallback threat assessment."""
        return {
            "overall_threat_level": "Medium",
            "key_threat_factors": ["Competitive market dynamics", "Resource constraints"],
            "mitigation_strategies": ["Strategic positioning", "Competitive monitoring"],
            "risk_matrix": {
                "high_impact_high_probability": [],
                "high_impact_low_probability": [],
                "low_impact_high_probability": [],
                "low_impact_low_probability": []
            }
        }

class RecommendationEngine:
    """Strategic recommendation generation."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def generate_recommendations(self, company_name: str, company_data: Dict,
                               analysis_results: Dict, user_context: Dict) -> List[Dict[str, Any]]:
        """Generate strategic recommendations."""
        try:
            with logging_context("strategic_recommendations", company=company_name):
                recommendations = []
                
                # Product strategy recommendations
                recommendations.append(self._generate_product_strategy(company_data, analysis_results, user_context))
                
                # Market strategy recommendations
                recommendations.append(self._generate_market_strategy(company_data, analysis_results, user_context))
                
                # Competitive strategy recommendations
                recommendations.append(self._generate_competitive_strategy(company_data, analysis_results, user_context))
                
                # Financial strategy recommendations (if enabled)
                if self.config.enable_financial_analysis:
                    recommendations.append(self._generate_financial_strategy(company_data, analysis_results, user_context))
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return self._fallback_recommendations()
    
    def _generate_product_strategy(self, company_data: Dict, analysis_results: Dict, 
                                 user_context: Dict) -> Dict[str, Any]:
        """Generate product strategy recommendations."""
        features = safe_get(company_data, "product_info", "key_features", default=[])
        
        recommendations = [
            "Conduct comprehensive competitive feature analysis to identify gaps",
            "Prioritize AI and automation capabilities for competitive differentiation",
            "Enhance mobile and API capabilities for broader platform integration",
            "Develop industry-specific features for vertical market penetration"
        ]
        
        # Add context-specific recommendations
        if len(features) < 5:
            recommendations.insert(0, "Expand core feature set to achieve competitive parity")
        
        return {
            "category": "Product Strategy",
            "priority": "High",
            "timeline": "6-12 months",
            "recommendations": recommendations[:4],
            "expected_impact": "Enhanced competitive positioning and customer retention"
        }
    
    def _generate_market_strategy(self, company_data: Dict, analysis_results: Dict,
                                user_context: Dict) -> Dict[str, Any]:
        """Generate market strategy recommendations."""
        recommendations = [
            "Target underserved market segments with tailored value propositions",
            "Expand geographic presence in markets with lower competitive intensity",
            "Develop strategic partnerships for channel expansion and capability enhancement",
            "Implement account-based marketing for high-value enterprise prospects"
        ]
        
        # Industry-specific recommendations
        industry = safe_get(user_context, "industry", "")
        if industry == "FinTech":
            recommendations.insert(0, "Develop compliance-focused features for regulated financial institutions")
        elif industry == "HealthTech":
            recommendations.insert(0, "Pursue healthcare-specific certifications and compliance frameworks")
        
        return {
            "category": "Market Strategy",
            "priority": "High",
            "timeline": "3-9 months",
            "recommendations": recommendations[:4],
            "expected_impact": "Market share growth and revenue diversification"
        }
    
    def _generate_competitive_strategy(self, company_data: Dict, analysis_results: Dict,
                                     user_context: Dict) -> Dict[str, Any]:
        """Generate competitive strategy recommendations."""
        return {
            "category": "Competitive Strategy",
            "priority": "Critical",
            "timeline": "1-3 months",
            "recommendations": [
                "Establish comprehensive competitive intelligence monitoring system",
                "Develop rapid response playbook for competitive threats and opportunities",
                "Create clear competitive differentiation messaging for sales and marketing",
                "Build meaningful switching costs through data integration and customization"
            ],
            "expected_impact": "Improved competitive win rates and reduced customer churn"
        }
    
    def _generate_financial_strategy(self, company_data: Dict, analysis_results: Dict,
                                   user_context: Dict) -> Dict[str, Any]:
        """Generate financial strategy recommendations."""
        return {
            "category": "Financial Strategy",
            "priority": "Medium",
            "timeline": "6-18 months",
            "recommendations": [
                "Optimize pricing strategy based on competitive positioning analysis",
                "Improve unit economics through customer acquisition cost optimization",
                "Develop multiple revenue streams to reduce dependency risks",
                "Plan funding strategy aligned with competitive positioning needs"
            ],
            "expected_impact": "Enhanced financial sustainability and competitive resilience"
        }
    
    def _fallback_recommendations(self) -> List[Dict[str, Any]]:
        """Fallback recommendations when generation fails."""
        return [{
            "category": "General Strategy",
            "priority": "Medium",
            "timeline": "3-6 months",
            "recommendations": ["Develop comprehensive competitive strategy"],
            "expected_impact": "Improved market position"
        }]

# ============================================================
# DELTA TRACKING
# ============================================================

class DeltaTracker:
    """Tracks changes between analysis runs."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def calculate_deltas(self, current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate differences between current and previous analysis."""
        if not previous or not self.config.enable_delta_tracking:
            return {"status": "no_previous_analysis"}
        
        try:
            deltas = {}
            
            # Compare market positioning
            deltas["market_positioning"] = self._compare_section(
                safe_get(current, "market_positioning", default={}),
                safe_get(previous, "market_positioning", default={})
            )
            
            # Compare SWOT analysis
            deltas["swot_analysis"] = self._compare_swot(
                safe_get(current, "swot_analysis", default={}),
                safe_get(previous, "swot_analysis", default={})
            )
            
            # Compare competitive landscape
            deltas["competitive_landscape"] = self._compare_section(
                safe_get(current, "competitive_landscape", default={}),
                safe_get(previous, "competitive_landscape", default={})
            )
            
            # Compare confidence scores
            current_confidence = safe_get(current, "confidence_score", 0)
            previous_confidence = safe_get(previous, "confidence_score", 0)
            deltas["confidence_change"] = round(current_confidence - previous_confidence, 3)
            
            # Calculate change summary
            deltas["summary"] = self._generate_change_summary(deltas)
            deltas["has_significant_changes"] = self._has_significant_changes(deltas)
            
            return deltas
            
        except Exception as e:
            logger.error(f"Delta calculation failed: {e}")
            return {"status": "delta_calculation_failed", "error": str(e)}
    
    def _compare_section(self, current: Dict, previous: Dict) -> Dict[str, Any]:
        """Compare two analysis sections."""
        changes = {}
        
        # Compare string fields
        for key in ["position", "segment", "growth_trajectory"]:
            curr_val = safe_get(current, key, "")
            prev_val = safe_get(previous, key, "")
            if curr_val != prev_val:
                changes[key] = {"from": prev_val, "to": curr_val}
        
        # Compare list fields
        for key in ["competitive_advantages", "direct_competitors", "competitive_moats"]:
            curr_list = safe_get(current, key, default=[])
            prev_list = safe_get(previous, key, default=[])
            
            added = [item for item in curr_list if item not in prev_list]
            removed = [item for item in prev_list if item not in curr_list]
            
            if added or removed:
                changes[key] = {"added": added, "removed": removed}
        
        return changes
    
    def _compare_swot(self, current: Dict, previous: Dict) -> Dict[str, Any]:
        """Compare SWOT analyses."""
        changes = {}
        
        for category in ["strengths", "weaknesses", "opportunities", "threats"]:
            curr_items = set(safe_get(current, category, default=[]))
            prev_items = set(safe_get(previous, category, default=[]))
            added = list(curr_items - prev_items)
            removed = list(prev_items - curr_items)
            
            if added or removed:
                changes[category] = {"added": added, "removed": removed}
        
        return changes
    
    def _generate_change_summary(self, deltas: Dict) -> List[str]:
        """Generate human-readable change summary."""
        summary = []
        
        # Market positioning changes
        mp_changes = deltas.get("market_positioning", {})
        if "position" in mp_changes:
            summary.append(f"Market position updated: {mp_changes['position']['from']} → {mp_changes['position']['to']}")
        
        # SWOT changes
        swot_changes = deltas.get("swot_analysis", {})
        for category, changes in swot_changes.items():
            if changes.get("added"):
                summary.append(f"New {category}: {len(changes['added'])} items added")
            if changes.get("removed"):
                summary.append(f"Updated {category}: {len(changes['removed'])} items changed")
        
        # Confidence changes
        confidence_change = deltas.get("confidence_change", 0)
        if abs(confidence_change) >= 0.1:
            direction = "increased" if confidence_change > 0 else "decreased"
            summary.append(f"Analysis confidence {direction} by {abs(confidence_change):.1%}")
        
        return summary[:5]  # Limit to 5 most important changes
    
    def _has_significant_changes(self, deltas: Dict) -> bool:
        """Determine if there are significant changes."""
        # Check for position changes
        if deltas.get("market_positioning", {}).get("position"):
            return True
        
        # Check for SWOT changes
        swot_changes = deltas.get("swot_analysis", {})
        total_swot_changes = sum(len(changes.get("added", [])) + len(changes.get("removed", [])) 
                               for changes in swot_changes.values())
        if total_swot_changes >= 3:
            return True
        
        # Check for significant confidence changes
        confidence_change = abs(deltas.get("confidence_change", 0))
        if confidence_change >= 0.15:
            return True
        
        return False

# ============================================================
# NEW: MULTI-MODEL ROUTER & ADAPTERS
# ============================================================

class _OpenAIAdapter:
    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int, timeout: float) -> str:
        """
        Tries both new and legacy OpenAI client shapes gracefully.
        Expected to return a string.
        """
        try:
            # Newer SDKs often expose client.chat.completions.create(...)
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a precise JSON generator."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            text = resp.choices[0].message.content
            return text
        except Exception as e1:
            try:
                # Some clients expose client.chat.completions.create without timeout kw or different signature
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "You are a precise JSON generator."},
                              {"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content
                return text
            except Exception:
                raise e1

class _AnthropicAdapter:
    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int, timeout: float) -> str:
        """
        Anthropic messages.create style.
        """
        try:
            resp = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout
            )
            # Unify content extraction
            if hasattr(resp, "content") and resp.content:
                block = resp.content[0]
                text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
                if not text and isinstance(resp.content, list):
                    parts = []
                    for b in resp.content:
                        t = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                        if t:
                            parts.append(t)
                    text = "\n".join(parts)
                return text or ""
            return ""
        except Exception as e:
            raise e

class MultiModelRouter:
    """
    Simple multi-model router that can call OpenAI or Anthropic
    based on per-stage preferences with auto-fallback.
    """
    def __init__(self, openai_client, anthropic_client, config: AnalysisConfig):
        self.config = config
        self.openai = _OpenAIAdapter(openai_client) if openai_client else None
        self.anthropic = _AnthropicAdapter(anthropic_client) if anthropic_client else None

    def available(self) -> bool:
        return bool(self.openai or self.anthropic)

    def generate(self, stage: str, prompt: str) -> Optional[str]:
        provider_pref = (self.config.llm_provider_per_stage or {}).get(stage, "auto")
        temperature = self.config.llm_temperature
        max_tokens = self.config.llm_max_tokens
        timeout = self.config.analysis_timeout_sec  # reuse analysis timeout

        # Helper to try a provider
        def try_openai():
            if not self.openai:
                return None
            return self.openai.generate(prompt, self.config.openai_model, temperature, max_tokens, timeout)

        def try_anthropic():
            if not self.anthropic:
                return None
            return self.anthropic.generate(prompt, self.config.anthropic_model, temperature, max_tokens, timeout)

        try:
            if provider_pref == "openai":
                return try_openai()
            if provider_pref == "anthropic":
                return try_anthropic()
            # auto: prefer OpenAI then Anthropic
            out = try_openai()
            if out:
                return out
            return try_anthropic()
        except Exception as e:
            logger.warning(f"LLM generation failed on {stage} with preferred={provider_pref}: {e}")
            # Fallback chain if first fails
            try:
                if provider_pref == "openai" and self.anthropic:
                    return try_anthropic()
                if provider_pref == "anthropic" and self.openai:
                    return try_openai()
            except Exception as e2:
                logger.warning(f"LLM fallback failed on {stage}: {e2}")
            return None

# ============================================================
# MAIN AI ANALYSIS ENGINE (FINAL CANONICAL DEFINITION)
# ============================================================

class AIAnalysisEngine:
    """
    Enhanced AI Analysis Engine with timeout handling, loop evaluation, 
    partial results, delta tracking, and standardized interfaces.
    """

    def __init__(
        self,
        openai_client=None,
        anthropic_client=None,
        config: Optional[AnalysisConfig] = None,
        ui_callbacks: Optional[UICallbacks] = None,
        analysis_callbacks: Optional[AnalysisCallbacks] = None
    ):
        """
        Initialize the AI Analysis Engine.
        """
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.config = config or AnalysisConfig()
        self.ui_callbacks = ui_callbacks
        self.analysis_callbacks = analysis_callbacks

        # --- Auto-initialize LLM clients from env if not provided ---
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

        if self.openai_client is None:
            if not OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not found. LLM features disabled; running in demo mode.")
                self.openai_client = None
            else:
                try:
                    self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    logger.info("✅ OpenAI client initialized from environment.")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    self.openai_client = None

        if self.anthropic_client is None and ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info("✅ Anthropic client initialized from environment.")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.anthropic_client = None
        # ------------------------------------------------------------

        # Initialize tools
        self.embedding_tool = EmbeddingTool(self.config)
        self.swot_analyzer = SWOTAnalyzer(self.config)
        self.competitive_analyzer = CompetitiveAnalyzer(self.config, self.embedding_tool)
        self.threat_analyzer = ThreatAnalyzer(self.config)
        self.recommendation_engine = RecommendationEngine(self.config)
        self.delta_tracker = DeltaTracker(self.config)

        # Multi-model router
        self.router = MultiModelRouter(self.openai_client, self.anthropic_client, self.config)

        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("AIAnalysisEngine initialized", extra={"config": asdict(self.config)})

    def analyze_competitor(
        self,
        company_name: str,
        company_data: Dict[str, Any],
        user_context: Dict[str, Any],
        previous_analysis: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Main analysis entry point with comprehensive error handling and fallbacks.
        """
        # Initialize result with metadata
        analysis_id = hashlib.md5(f"{company_name}-{datetime.now().isoformat()}".encode()).hexdigest()
        metadata = AnalysisMetadata(
            analysis_id=analysis_id,
            started_at=datetime.now()
        )
        
        result = AnalysisResult(
            competitor_name=company_name,
            metadata=metadata
        )
        
        try:
            with logging_context("full_analysis", company=company_name, analysis_id=analysis_id):
                self._update_progress(10, "Initializing analysis...")
                
                # Execute analysis with loop and evaluation
                for loop_num in range(self.config.max_plan_loops):
                    metadata.loops_executed = loop_num + 1
                    
                    try:
                        # Execute analysis stages
                        result = self._execute_analysis_stages(company_name, company_data, user_context, result)
                        
                        # Check if we meet confidence threshold
                        if result.confidence_score >= self.config.confidence_threshold:
                            logger.info(f"Analysis meets confidence threshold after {loop_num + 1} loops")
                            break
                        
                        # If not final loop and retry enabled, continue
                        if loop_num < self.config.max_plan_loops - 1 and self.config.retry_on_low_confidence:
                            logger.info(f"Confidence {result.confidence_score:.2f} below threshold {self.config.confidence_threshold:.2f}, retrying...")
                            self._update_progress(30 + (loop_num * 20), f"Refining analysis (attempt {loop_num + 2})...")
                        
                    except Exception as e:
                        logger.error(f"Analysis loop {loop_num + 1} failed: {e}")
                        if loop_num == self.config.max_plan_loops - 1:
                            raise  # Re-raise on final attempt
                
                # Calculate deltas if previous analysis provided
                if previous_analysis and self.config.enable_delta_tracking:
                    self._update_progress(85, "Calculating changes from previous analysis...")
                    result.changes_from_previous = self.delta_tracker.calculate_deltas(
                        result.to_dict(), previous_analysis
                    )
                
                # Finalize result
                metadata.completed_at = datetime.now()
                metadata.duration_sec = (metadata.completed_at - metadata.started_at).total_seconds()
                result.status = "completed"
                
                self._update_progress(100, "Analysis complete")
                self._notify_done()
                
                logger.info("Analysis completed successfully", extra={
                    "company": company_name,
                    "confidence": result.confidence_score,
                    "duration": metadata.duration_sec,
                    "loops": metadata.loops_executed
                })
                
                return result
                
        except Exception as e:
            logger.error(f"Analysis failed for {company_name}: {e}")
            
            # Return partial results if enabled
            if self.config.return_partial_on_failure and self._has_minimal_results(result):
                result.status = "partial"
                result.metadata.errors.append(str(e))
                result.metadata.completed_at = datetime.now()
                result.metadata.duration_sec = (result.metadata.completed_at - result.metadata.started_at).total_seconds()
                
                logger.info("Returning partial results after failure", extra={
                    "company": company_name,
                    "error": str(e)
                })
                
                self._update_progress(100, "Analysis completed with partial results")
                self._notify_done()
                
                return result
            
            # Return fallback result
            result = self._create_fallback_result(company_name, str(e), metadata)
            
            self._update_progress(100, "Analysis failed - returned fallback results")
            self._notify_done()
            
            return result
        
        finally:
            # Clean up executor
            if hasattr(self, 'executor'):
                try:
                    self.executor.shutdown(wait=False)
                except Exception:
                    pass
    
    def _execute_analysis_stages(
        self,
        company_name: str,
        company_data: Dict[str, Any],
        user_context: Dict[str, Any],
        result: AnalysisResult
    ) -> AnalysisResult:
        """Execute all analysis stages with timeout handling."""
        
        stages = [
            ("swot_analysis", self._execute_swot_analysis, 20),
            ("market_positioning", self._execute_market_positioning, 35),
            ("competitive_landscape", self._execute_competitive_landscape, 50),
            ("strategic_recommendations", self._execute_strategic_recommendations, 70),
            ("threat_assessment", self._execute_threat_assessment, 85)
        ]
        
        for stage_name, stage_func, progress_pct in stages:
            try:
                self._update_progress(progress_pct, f"Executing {stage_name.replace('_', ' ')}...")
                
                # Execute with timeout
                stage_result = self._execute_with_timeout(
                    stage_func,
                    self.config.analysis_timeout_sec,
                    company_name,
                    company_data,
                    user_context,
                    result
                )

                # ---------------- NEW: optional LLM JSON refinement per stage ----------------
                if self.config.enable_llm_refinement and self.router.available() and stage_result:
                    try:
                        stage_result = self._maybe_llm_refine(
                            stage_name=stage_name,
                            stage_result=stage_result,
                            company_name=company_name,
                            user_context=user_context
                        )
                    except Exception as e:
                        logger.warning(f"LLM refinement skipped on {stage_name}: {e}")
                # ---------------------------------------------------------------------------

                # Update result based on stage
                self._update_result_with_stage(result, stage_name, stage_result)
                result.metadata.stages_completed.append(stage_name)
                
                if self.analysis_callbacks:
                    self.analysis_callbacks.on_stage_complete(stage_name, stage_result, 0.0)
                
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                result.metadata.stages_failed.append(stage_name)
                result.metadata.errors.append(f"{stage_name}: {str(e)}")
                
                if self.analysis_callbacks:
                    self.analysis_callbacks.on_error(stage_name, e)
                
                # Continue with other stages unless critical failure
                if not self.config.graceful_degradation and stage_name in self.config.min_required_sections:
                    raise
        
        # Calculate overall confidence
        result.confidence_score = self._calculate_overall_confidence(result)
        
        return result
    
    def _execute_with_timeout(self, func: Callable, timeout_sec: float, *args, **kwargs):
        """Execute function with timeout using thread pool."""
        if not hasattr(self, 'executor') or self.executor is None:
            # Fallback to direct execution if executor is not available
            return func(*args, **kwargs)
        
        try:
            future = self.executor.submit(func, *args, **kwargs)
            return future.result(timeout=timeout_sec)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(f"Operation timed out after {timeout_sec} seconds")
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                # Fallback to direct execution if executor is shutdown
                return func(*args, **kwargs)
            raise
    
    def _execute_swot_analysis(self, company_name: str, company_data: Dict, user_context: Dict, result: AnalysisResult) -> Dict[str, Any]:
        """Execute SWOT analysis stage."""
        if not self.config.enable_swot_analysis:
            return {}
        
        return self.swot_analyzer.analyze(company_name, company_data, user_context)
    
    def _execute_market_positioning(self, company_name: str, company_data: Dict, user_context: Dict, result: AnalysisResult) -> Dict[str, Any]:
        """Execute market positioning stage."""
        return self.competitive_analyzer.analyze_positioning(company_name, company_data, user_context)
    
    def _execute_competitive_landscape(self, company_name: str, company_data: Dict, user_context: Dict, result: AnalysisResult) -> Dict[str, Any]:
        """Execute competitive landscape analysis stage."""
        if not self.config.enable_competitive_analysis:
            return {}
        
        return self.competitive_analyzer.analyze_landscape(company_name, company_data, user_context)
    
    def _execute_strategic_recommendations(self, company_name: str, company_data: Dict, user_context: Dict, result: AnalysisResult) -> List[Dict[str, Any]]:
        """Execute strategic recommendations stage."""
        return self.recommendation_engine.generate_recommendations(
            company_name, company_data, result.to_dict(), user_context
        )
    
    def _execute_threat_assessment(self, company_name: str, company_data: Dict, user_context: Dict, result: AnalysisResult) -> Dict[str, Any]:
        """Execute threat assessment stage."""
        return self.threat_analyzer.analyze_threats(
            company_name, company_data, result.to_dict(), user_context
        )
    
    def _update_result_with_stage(self, result: AnalysisResult, stage_name: str, stage_result: Any):
        """Update result object with stage output."""
        if stage_name == "swot_analysis":
            result.swot_analysis = stage_result
            result.section_confidences["swot_analysis"] = stage_result.get("analysis_confidence", 0.7)
        elif stage_name == "market_positioning":
            result.market_positioning = stage_result
            result.section_confidences["market_positioning"] = 0.8  # Default confidence
        elif stage_name == "competitive_landscape":
            result.competitive_landscape = stage_result
            result.section_confidences["competitive_landscape"] = 0.75
        elif stage_name == "strategic_recommendations":
            result.strategic_recommendations = stage_result
            result.section_confidences["strategic_recommendations"] = 0.7
        elif stage_name == "threat_assessment":
            result.threat_assessment = stage_result
            result.section_confidences["threat_assessment"] = 0.8
    
    def _calculate_overall_confidence(self, result: AnalysisResult) -> float:
        """Calculate overall confidence score."""
        section_scores = list(result.section_confidences.values())
        if not section_scores:
            return 0.5
        
        return calculate_confidence(section_scores)
    
    def _has_minimal_results(self, result: AnalysisResult) -> bool:
        """Check if result has minimal required content."""
        required_sections = self.config.min_required_sections
        for section in required_sections:
            if section == "market_positioning" and not result.market_positioning:
                return False
            if section == "swot_analysis" and not result.swot_analysis:
                return False
        
        return True
    
    def _create_fallback_result(self, company_name: str, error_msg: str, metadata: AnalysisMetadata) -> AnalysisResult:
        """Create fallback result when analysis fails."""
        metadata.completed_at = datetime.now()
        metadata.duration_sec = (metadata.completed_at - metadata.started_at).total_seconds()
        metadata.errors.append(error_msg)
        
        return AnalysisResult(
            competitor_name=company_name,
            market_positioning={
                "segment": "Technology Software",
                "position": "Analysis failed - data insufficient",
                "market_share": "Unable to determine",
                "growth_trajectory": "Assessment pending",
                "competitive_advantages": ["Requires additional analysis"]
            },
            swot_analysis={
                "strengths": [f"{company_name} has established market presence"],
                "weaknesses": ["Limited data available for analysis"],
                "opportunities": ["Market analysis pending"],
                "threats": ["Competitive assessment required"],
                "analysis_confidence": 0.2
            },
            competitive_landscape={
                "direct_competitors": ["Analysis required"],
                "indirect_competitors": ["Assessment pending"],
                "competitive_moats": ["Evaluation needed"],
                "vulnerability_areas": ["Analysis in progress"]
            },
            strategic_recommendations=[{
                "category": "Data Collection",
                "priority": "Critical",
                "timeline": "Immediate",
                "recommendations": ["Gather additional competitive intelligence"],
                "expected_impact": "Enable comprehensive analysis"
            }],
            threat_assessment={
                "overall_threat_level": "Unknown",
                "key_threat_factors": ["Insufficient data for assessment"],
                "mitigation_strategies": ["Complete data collection and re-analyze"]
            },
            confidence_score=0.2,
            status="failed",
            metadata=metadata
        )
    
    def _update_progress(self, pct: int, message: str):
        """Update progress via UI callbacks."""
        if self.ui_callbacks:
            try:
                self.ui_callbacks.on_progress(pct, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _notify_done(self):
        """Notify completion via UI callbacks."""
        if self.ui_callbacks:
            try:
                self.ui_callbacks.on_done()
            except Exception as e:
                logger.warning(f"Done callback failed: {e}")

    # ---------------- NEW: LLM refinement helper ----------------
    def _maybe_llm_refine(
        self,
        stage_name: str,
        stage_result: Any,
        company_name: str,
        user_context: Dict[str, Any],
    ) -> Any:
        """
        Ask the chosen LLM to lightly refine and structure the stage_result.
        Returns the original stage_result on any failure.
        """
        # Only refine dicts/lists we can round-trip as JSON
        if not isinstance(stage_result, (dict, list)):
            return stage_result

        # Build a strict JSON-only prompt
        schema_hint = {
            "stage": stage_name,
            "company_name": company_name,
            "expected_type": "object" if isinstance(stage_result, dict) else "array",
            "keep_fields": True,
            "no_new_unverifiable_facts": True,
            "fill_small_gaps": True,
            "return_json_only": True
        }

        user_payload = {
            "instructions": (
                "You are given a JSON payload produced by a rules-based analyzer. "
                "Lightly improve clarity, deduplicate items, tighten phrasing, and (only if safe) "
                "merge overlapping points. Keep the same schema and keys. "
                "Do NOT invent new facts. If unsure, keep the original text. "
                "Return ONLY valid, minified JSON—no prose."
            ),
            "context": {
                "company_name": company_name,
                "user_context": user_context
            },
            "schema_hint": schema_hint,
            "input_json": stage_result
        }

        prompt = json.dumps(user_payload, ensure_ascii=False)

        if not self.router.available():
            return stage_result

        llm_text = self.router.generate(stage=stage_name, prompt=prompt)
        if not llm_text:
            return stage_result

        # Try to locate JSON if the model added any stray text
        try:
            # Fast path: already pure JSON
            refined = json.loads(llm_text)
            return refined
        except Exception:
            try:
                start = llm_text.find("{") if isinstance(stage_result, dict) else llm_text.find("[")
                end = llm_text.rfind("}") if isinstance(stage_result, dict) else llm_text.rfind("]")
                if 0 <= start < end:
                    refined = json.loads(llm_text[start:end+1])
                    return refined
            except Exception as e2:
                logger.warning(f"Failed to parse LLM JSON for {stage_name}: {e2}")
            return stage_result
    # -----------------------------------------------------------

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'executor') and self.executor is not None:
            try:
                self.executor.shutdown(wait=True)
            except Exception:
                pass

# ============================================================
# PRESERVED ORIGINAL/STRAY/DUPE BLOCKS (COMMENTED OUT, NOT REMOVED)
# ============================================================

# The following blocks were present in your source multiple times or at top-level using `self`
# and would break execution. They are preserved verbatim but commented out.

"""
# --- Auto-initialize LLM clients from env if not provided ---
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# OpenAI
if self.openai_client is None:
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found. LLM features (OpenAI) will be disabled; running in demo/rule-based mode.")
        self.openai_client = None
    else:
        try:
            from openai import OpenAI  # openai>=1.x
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized from environment.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None

# Anthropic (optional)
if self.anthropic_client is None and ANTHROPIC_API_KEY:
    try:
        import anthropic
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized from environment.")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        self.anthropic_client = None
"""

"""
class AIAnalysisEngine:
    def __init__(self, openai_client=None, anthropic_client=None, config=None, ui_callbacks=None, analysis_callbacks=None):
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.config = config or AnalysisConfig()
        self.ui_callbacks = ui_callbacks
        self.analysis_callbacks = analysis_callbacks

        # >>> Added block
        if self.openai_client is None:
            if not OPENAI_API_KEY:
                logging.warning("OPENAI_API_KEY not found. Running in demo mode.")
                self.openai_client = None
            else:
                try:
                    self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    logging.info("✅ OpenAI client initialized from .env")
                except Exception as e:
                    logging.error(f"Failed to initialize OpenAI client: {e}")
                    self.openai_client = None
"""

"""
# DUPLICATE AIAnalysisEngine class header block (already defined above):
class AIAnalysisEngine:
    """
# (rest omitted intentionally; canonical definition retained above)


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_analysis_engine(
    openai_client=None,
    anthropic_client=None,
    config: Optional[AnalysisConfig] = None,
    ui_callbacks: Optional[UICallbacks] = None,
    analysis_callbacks: Optional[AnalysisCallbacks] = None
) -> AIAnalysisEngine:
    """
    Factory function to create a configured AI Analysis Engine.

    Args:
        openai_client: Optional OpenAI client
        anthropic_client: Optional Anthropic client
        config: Analysis configuration
        ui_callbacks: UI progress callbacks
        analysis_callbacks: Analysis event callbacks

    Returns:
        Configured AIAnalysisEngine instance
    """
    if config is None:
        config = AnalysisConfig()

    engine = AIAnalysisEngine(
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        config=config,
        ui_callbacks=ui_callbacks,
        analysis_callbacks=analysis_callbacks
    )

    logger.info("Created AI Analysis Engine via factory", extra={"config": asdict(config)})
    return engine


# ============================================================
# VALIDATION AND TESTING
# ============================================================

def validate_analysis_result(result: AnalysisResult) -> Tuple[bool, List[str]]:
    """
    Validate that analysis result meets quality standards.

    Args:
        result: AnalysisResult to validate

    Returns:
        Tuple containing:
        - bool: True if result is valid, False otherwise
        - List[str]: List of validation issues found
    """
    issues = []

    # Check required sections
    if not result.market_positioning:
        issues.append("Missing market positioning analysis")

    if not result.swot_analysis:
        issues.append("Missing SWOT analysis")
    elif not all(k in result.swot_analysis for k in ["strengths", "weaknesses", "opportunities", "threats"]):
        issues.append("Incomplete SWOT analysis sections")

    if not result.strategic_recommendations:
        issues.append("Missing strategic recommendations")

    # Check confidence scores
    if result.confidence_score < 0.3:
        issues.append(f"Very low confidence score: {result.confidence_score:.2f}")

    # Check metadata
    if not result.metadata.analysis_id:
        issues.append("Missing analysis ID")

    if result.status not in ["completed", "partial", "failed"]:
        issues.append(f"Invalid status: {result.status}")

    return len(issues) == 0, issues


# Export main classes and functions
__all__ = [
    "AIAnalysisEngine",
    "AnalysisConfig",
    "AnalysisResult",
    "AnalysisMetadata",
    "UICallbacks",
    "AnalysisCallbacks",
    "create_analysis_engine",
    "validate_analysis_result"
]
