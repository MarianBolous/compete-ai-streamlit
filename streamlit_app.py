# streamlit_app.py
"""
CompeteAI Platform - Main Streamlit Application
===============================================

Advanced AI-Powered Competitive Intelligence & Strategic Analysis Platform
Multi-Agent Architecture with Production-Ready Orchestration

Usage:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

# External imports
import nest_asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Apply nest_asyncio to handle async in Streamlit
try:
    nest_asyncio.apply()
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Import version
from version import __version__ as app_version

# Environment variables
SERVER_KEY = os.getenv("OPENAI_API_KEY", "")
ALLOW_BYOK = os.getenv("ALLOW_BYOK", "true").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# ----------------------------- 
# FastAPI Application Setup
# ----------------------------- 

app = FastAPI()

class AnalyzeRequest(BaseModel):
    prompt: str

def pick_key(user_key_header: str | None) -> tuple[str | None, str]:
    if ALLOW_BYOK and user_key_header and user_key_header.startswith("sk-"):
        return user_key_header, "user"
    if SERVER_KEY.startswith("sk-"):
        return SERVER_KEY, "server"
    if DEMO_MODE:
        return None, "demo"
    return None, "none"

@app.post("/analyze")
def analyze(req: AnalyzeRequest, x_user_openai_key: str | None = Header(default=None)):
    key, source = pick_key(x_user_openai_key)

    if source == "demo":
        return {"provider": "demo", "model": "mock-gpt", "content": f"[DEMO] {req.prompt[:80]}...", "confidence": 0.65}

    if not key:
        raise HTTPException(status_code=400, detail="No API key available. Enable server key or BYOK, or use demo mode.")

    headers = {"Authorization": f"Bearer {key}"}
    payload = {"model": "gpt-4o-mini", "input": req.prompt}
    r = requests.post("https://api.openai.com/v1/responses", json=payload, headers=headers, timeout=45)
    r.raise_for_status()
    data = r.json()
    text = (data.get("output", {}) or data.get("content") or data).__str__()
    return {"provider": "openai", "model": payload["model"], "content": text[:4000], "confidence": 0.8}

# ----------------------------- 
# Logo and Asset Paths
# ----------------------------- 

import os
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
LOGO_PATH = os.path.join(ASSETS_DIR, 'logo.svg')
# -----------------------------
# Page Configuration
# -----------------------------

# Load SVG logo for page icon
with open(LOGO_PATH, "r") as f:
    logo_svg = f.read()

st.set_page_config(
    page_title="CompeteAI - Advanced Competitive Intelligence",
    page_icon=logo_svg,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS Styling
# -----------------------------

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-top: 10px;
    }
    
    /* Enhanced metric containers with animations */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    /* Enhanced analysis cards */
    .analysis-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-left: 4px solid #1f77b4;
    }
    
    .analysis-card:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
        border-left: 4px solid #ff7f0e;
    }
    
    .threat-high { 
        background: linear-gradient(135deg, #dc3545, #c82333); 
        color: white;
    }
    .threat-medium { 
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: black; 
    }
    .threat-low { 
        background: linear-gradient(135deg, #28a745, #1e7e34);
        color: white; 
    }
    
    /* Enhanced notification styling */
    .notification {
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        position: relative;
        overflow: hidden;
    }
    
    .notification::before {
        content: '';
        position: absolute;
        top: 0;
    
    /* Modern Compact BYOK Section Styling */
    .byok-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        color: white;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .byok-section:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.16);
    }
    
    .byok-title {
        font-size: 1.05rem;
        font-weight: bold;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Centered BYOK text with individual brand colors for each letter */
    .byok-section {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .byok-title {
        font-size: 1.5rem;
        font-weight: 900;
        padding: 8px 0;
        display: flex;
        justify-content: center;
        gap: 4px;
        margin: 0 auto;
        width: fit-content;
    }
    
    /* Individual colors for each letter using brand colors */
    .byok-letter-b {
        color: #1f77b4; /* Blue */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        transform: scale(1.1);
        display: inline-block;
    }
    
    .byok-letter-y {
        color: #ff7f0e; /* Orange */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        transform: scale(1.1);
        display: inline-block;
    }
    
    .byok-letter-o {
        color: #2ca02c; /* Green */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        transform: scale(1.1);
        display: inline-block;
    }
    
    .byok-letter-k {
        color: #9467bd; /* Purple */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        transform: scale(1.1);
        display: inline-block;
    }
    
    /* Optional hover effects for each letter */
    .byok-title span:hover {
        transform: scale(1.2) translateY(-2px);
        transition: all 0.3s ease;
    }
    
    .byok-description {
        font-size: 0.85rem;
        opacity: 0.95;
        margin-bottom: 6px;
        line-height: 1.3;
        font-weight: 500;
    }
    
    .byok-badge {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 1px 8px;
        font-size: 0.7rem;
        font-weight: 500;
        display: inline-block;
    }
    
    .server-keys-badge {
        background: rgba(255,255,255,0.9);
        color: #495057;
        border-radius: 10px;
        padding: 1px 8px;
        font-size: 0.7rem;
        font-weight: 500;
        display: inline-block;
    }
        left: 0;
        width: 5px;
        height: 100%;
        background: rgba(255,255,255,0.3);
    }
    
    .success-notification {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
    }
    
    .error-notification {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
    }
    
    .warning-notification {
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: black;
    }
    
    /* Enhanced sidebar styling */
    .sidebar-logo {
        text-align: center;
        padding: 20px;
        font-size: 2rem;
        font-weight: bold;
        color: white;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        margin-bottom: 20px;
    }
    
    /* Add subtle hover effect to sidebar image */
    .sidebar img:hover {
        transform: scale(1.05);
        transition: transform 0.3s ease;
    }
    
    /* Make sidebar more blue according to brand color */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f77b4 0%, #1565c0 100%);
        color: white;
    }
    
    /* Style sidebar headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6, 
    [data-testid="stSidebar"] strong {
        color: white !important;
    }
    
    /* Style sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Style sidebar separators */
    [data-testid="stSidebar"] hr {
        background-color: rgba(255,255,255,0.2);
    }
    
    /* Enhanced progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        background-size: 200% 100%;
        animation: progress-bar-animation 2s infinite linear;
    }
    
    @keyframes progress-bar-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Style for sidebar expander - make it static blue whether expanded or collapsed */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: rgba(0, 40, 80, 0.6);
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 10px;
    }
    
    /* Style for sidebar expander content area */
    [data-testid="stSidebar"] [data-testid="stExpander"] > div:nth-child(2) {
        background-color: transparent;
        padding: 10px 5px;
        margin-top: 5px;
    }
    
    /* Style for sidebar expander header */
    [data-testid="stSidebar"] [data-testid="stExpander"] > div:nth-child(1) {
        color: white !important;
    }
    
    /* Style for inputs inside sidebar expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] input[type="text"],
    [data-testid="stSidebar"] [data-testid="stExpander"] input[type="password"] {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
    }
    
    /* Style for input placeholders inside sidebar expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Data Classes
# -----------------------------

@dataclass
class CompetitorAnalysis:
    """Main analysis result dataclass used by UI"""
    competitor_name: str
    company_info: Dict[str, Any]
    market_analysis: Dict[str, Any]
    strategic_insights: List[Dict[str, Any]]
    confidence_score: float
    generated_at: str
    status: Optional[str] = "ok"
    warnings: Optional[List[str]] = None
    diff: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

# -----------------------------
# Agent Module Imports
# -----------------------------

# Track import status
AGENTS_OK = True
import_errors = []

try:
    from agents.data_collection_agent import DataCollectionAgent, create_data_collection_agent
    logger.info("✅ DataCollectionAgent imported successfully")
except ImportError as e:
    AGENTS_OK = False
    import_errors.append(f"DataCollectionAgent: {e}")
    logger.error(f"Failed to import DataCollectionAgent: {e}")
    
    # Fallback mock for demo
    class DataCollectionAgent:
        def gather_company_data(self, company_name: str) -> Dict[str, Any]:
            return {
                "company_overview": {"name": company_name, "industry": "Technology"},
                "product_info": {"main_product": f"{company_name} Platform"},
                "financial_data": {},
                "recent_news": [],
            }

try:
    from agents.analysis_engine import AIAnalysisEngine
    logger.info("✅ AIAnalysisEngine imported successfully")
except ImportError as e:
    AGENTS_OK = False
    import_errors.append(f"AIAnalysisEngine: {e}")
    logger.error(f"Failed to import AIAnalysisEngine: {e}")
    
    # Utility function for safe dictionary navigation
    def safe_get(data: Dict, *keys, default=None):
        """Safely navigate nested dictionaries."""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    # Fallback mock for demo
    class AIAnalysisEngine:
        def __init__(self, openai_client=None, anthropic_client=None):
            self.openai_client = openai_client
            self.anthropic_client = anthropic_client
            
        def analyze_competitor(self, company_name: str, company_data: Dict, user_context: Dict) -> Dict:
            # Assess threat level
            threat_level = self._assess_overall_threat_level(company_data, {})
            
            return {
                "market_positioning": {"segment": "B2B SaaS", "position": "Growing"},
                "swot_analysis": {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []},
                "strategic_recommendations": [],
                "confidence_score": 0.75,
                "threat_assessment": {
                    "overall_threat_level": threat_level,
                    "key_threat_factors": []
                }
            }
            
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

try:
    from agents.orchestrator import CompeteAIOrchestrator, OrchestratorConfig
    logger.info("✅ CompeteAIOrchestrator imported successfully")
except ImportError as e:
    AGENTS_OK = False
    import_errors.append(f"CompeteAIOrchestrator: {e}")
    logger.error(f"Failed to import CompeteAIOrchestrator: {e}")
    
    # Fallback mock for demo
    class OrchestratorConfig:
        def __init__(self, **kwargs):
            self.max_plan_loops = kwargs.get('max_plan_loops', 2)
            self.confidence_threshold = kwargs.get('confidence_threshold', 0.7)
            self.timeout_collect_sec = kwargs.get('timeout_collect_sec', 30)
            self.timeout_analyze_sec = kwargs.get('timeout_analyze_sec', 40)
            self.simulated_delays = kwargs.get('simulated_delays', False)
    
    class CompeteAIOrchestrator:
        def __init__(self, **kwargs):
            pass
        def run_analysis(self, competitor_name: str, user_context: Dict, **kwargs) -> Dict:
            return {
                "competitor_name": competitor_name,
                "company_info": {},
                "market_analysis": {},
                "strategic_insights": [],
                "confidence_score": 0.75,
                "generated_at": datetime.now().isoformat()
            }

# -----------------------------
# Utility Functions
# -----------------------------

class StreamlitUICallbacks:
    """Callbacks for orchestrator progress updates"""
    
    def __init__(self, progress_bar, status_text):
        self.progress_bar = progress_bar
        self.status_text = status_text
        
    def on_progress(self, percent: int, message: str) -> None:
        """Update progress bar and status text"""
        try:
            self.status_text.text(message)
            self.progress_bar.progress(min(100, max(0, int(percent))) / 100)
        except Exception as e:
            logger.debug(f"Progress update error: {e}")
    
    def on_done(self) -> None:
        """Clear progress indicators"""
        try:
            self.status_text.empty()
            self.progress_bar.empty()
        except Exception as e:
            logger.debug(f"Clear progress error: {e}")
    
    def on_error(self, error: str) -> None:
        """Display error message"""
        try:
            self.status_text.error(f"Error: {error}")
        except Exception as e:
            logger.debug(f"Error display error: {e}")

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'last_result_dict' not in st.session_state:
        st.session_state.last_result_dict = None
    
    if 'api_keys_validated' not in st.session_state:
        st.session_state.api_keys_validated = False

# -----------------------------
# Sidebar Configuration
# -----------------------------

def setup_sidebar() -> Dict[str, Any]:
    """Setup sidebar configuration and return user context"""
    
    # Logo and title
    st.sidebar.image(LOGO_PATH, width=120)
    st.sidebar.markdown("**CompeteAI**")
    st.sidebar.markdown("Advanced Competitive Intelligence Platform")
    
    # Display import status
    if not AGENTS_OK:
        st.sidebar.warning("⚠️ Some modules failed to import. Running in limited mode.")
        with st.sidebar.expander("Import Issues"):
            for error in import_errors:
                st.error(error)
    
    # API Configuration
    st.sidebar.markdown("### 🤖 AI Configuration")
    
    # Show BYOK status with modern styling
    if ALLOW_BYOK:
        st.sidebar.markdown("""
        <div class="byok-section">
            <div class="byok-title">
                <span class="byok-letter-b">B</span>
                <span class="byok-letter-y">Y</span>
                <span class="byok-letter-o">O</span>
                <span class="byok-letter-k">K</span>
            </div>
            <p class="byok-description">Bring Your Own Key</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="byok-section">
            <div class="byok-title">🔑 API Key Mode</div>
            <p class="byok-description">Using server-provided API keys for analysis operations.</p>
            <span class="server-keys-badge">SERVER KEYS</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo mode checkbox
    demo_mode = st.sidebar.checkbox(
        "Demo Mode",
        value=DEMO_MODE,
        help="Use realistic mock data for demonstration purposes"
    )
    
    # API Key input fields with modern styling
    with st.sidebar.expander("🔐 API Keys", expanded=ALLOW_BYOK):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for analysis" + (" (BYOK enabled)" if ALLOW_BYOK else ""),
            placeholder="sk-...",
            disabled=not ALLOW_BYOK and SERVER_KEY.startswith("sk-")
        )
        
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Optional - Enter your Anthropic API key for enhanced AI analysis",
            placeholder="sk-ant-..."
        )
        
        serpapi_key = st.text_input(
            "SERPAPI Key",
            type="password",
            help="Optional - Enter your SERPAPI key for enhanced data collection",
            placeholder="Your SERPAPI key"
        )
    
    # Initialize AI clients
    openai_client = None
    anthropic_client = None
    
    # Use user key if provided, otherwise fall back to server key
    active_openai_key = openai_key if openai_key else SERVER_KEY
    
    if active_openai_key and not demo_mode:
        try:
            import openai
            openai_client = openai.OpenAI(api_key=active_openai_key)
            key_source = "user" if openai_key else "server"
            st.sidebar.success(f"✅ OpenAI connected (using {key_source} key)")
        except Exception as e:
            st.sidebar.error(f"❌ OpenAI connection failed: {str(e)}")
    
    if anthropic_key and not demo_mode:
        try:
            from anthropic import Anthropic
            anthropic_client = Anthropic(api_key=anthropic_key)
            st.sidebar.success("✅ Anthropic connected")
        except Exception as e:
            st.sidebar.error(f"❌ Anthropic connection failed: {str(e)}")
    
    if demo_mode:
        st.sidebar.info("📊 Demo mode active - using realistic mock data")
    
    st.sidebar.markdown("---")
    
    # User Context
    st.sidebar.markdown("### 🏢 Your Company Context")
    
    company_name = st.sidebar.text_input(
        "Company Name",
        value="YourStartup Inc.",
        help="Enter your company name for personalized analysis"
    )
    
    industry = st.sidebar.selectbox(
        "Industry",
        ["SaaS", "E-commerce", "FinTech", "HealthTech", "EdTech", 
         "MarTech", "AI/ML", "Enterprise Software", "Other"],
        help="Select your industry for relevant competitive insights"
    )
    
    company_stage = st.sidebar.selectbox(
        "Company Stage",
        ["Idea", "MVP", "Early Stage", "Growth Stage", "Scale Stage", "Mature"],
        index=2,
        help="Select your company's current stage"
    )
    
    team_size = st.sidebar.selectbox(
        "Team Size",
        ["1-10", "11-50", "51-200", "201-1000", "1000+"],
        index=1,
        help="Select your approximate team size"
    )
    
    company_focus = st.sidebar.text_input(
        "Strategic Focus",
        value="product innovation and market expansion",
        help="Describe your current strategic focus"
    )
    
    st.sidebar.markdown("---")
    
    # Analysis Options
    st.sidebar.markdown("### ⚙️ Analysis Options")
    
    analysis_depth = st.sidebar.selectbox(
        "Analysis Depth",
        ["Quick Overview", "Standard Analysis", "Deep Dive"],
        index=1,
        help="Select the depth of competitive analysis"
    )
    
    include_financials = st.sidebar.checkbox(
        "Include Financial Analysis",
        value=True,
        help="Include funding, revenue, and financial health analysis"
    )
    
    enable_iterations = st.sidebar.checkbox(
        "Enable Smart Iterations",
        value=True,
        help="Allow the system to iterate for better results"
    )
    
    max_iterations = st.sidebar.slider(
        "Max Iterations",
        min_value=1,
        max_value=5,
        value=2,
        help="Maximum analysis iterations"
    ) if enable_iterations else 1
    
    st.sidebar.markdown("---")
    
    # Advanced Settings
    with st.sidebar.expander("Advanced Settings"):
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Minimum confidence score for analysis"
        )
        
        enable_caching = st.sidebar.checkbox(
            "Enable Caching",
            value=True,
            help="Cache results for faster repeated analyses"
        )
        
        timeout_seconds = st.sidebar.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=120,
            value=30,
            help="Maximum time for each agent operation"
        )
    
    return {
        "openai_client": openai_client,
        "anthropic_client": anthropic_client,
        "serpapi_key": serpapi_key,
        "demo_mode": demo_mode,
        "company_name": company_name,
        "company_focus": company_focus,
        "industry": industry,
        "company_stage": company_stage,
        "team_size": team_size,
        "analysis_depth": analysis_depth,
        "include_financials": include_financials,
        "max_iterations": max_iterations,
        "confidence_threshold": confidence_threshold,
        "enable_caching": enable_caching,
        "timeout_seconds": timeout_seconds,
    }

# -----------------------------
# Analysis Display Functions
# -----------------------------

def display_analysis_results(analysis: CompetitorAnalysis):
    """Display comprehensive analysis results"""
    
    # Header with key metrics
    st.markdown(
        f'<h1 class="main-header">Analysis Report: {analysis.competitor_name}</h1>',
        unsafe_allow_html=True
    )
    
    # Display warnings if any
    if analysis.warnings:
        for warning in analysis.warnings:
            st.warning(warning)
    
    # Executive summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    market_analysis = analysis.market_analysis
    positioning = market_analysis.get('market_positioning', {})
    threat_assessment = market_analysis.get('threat_assessment', {})
    
    with col1:
        position_text = positioning.get('position', 'Growing Player')
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Market Position</h3>
            <p>{position_text[:30]}...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        threat_level = threat_assessment.get('overall_threat_level', 'Medium')
        threat_class = (
            "threat-high" if "High" in threat_level
            else "threat-medium" if "Medium" in threat_level
            else "threat-low"
        )
        st.markdown(f"""
        <div class="metric-container {threat_class}">
            <h3>⚠️ Threat Level</h3>
            <p>{threat_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = analysis.confidence_score
        st.markdown(f"""
        <div class="metric-container">
            <h3>🎯 Confidence</h3>
            <p>{confidence*100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        growth = positioning.get('growth_trajectory', 'Strong Growth')
        st.markdown(f"""
        <div class="metric-container">
            <h3>📈 Growth</h3>
            <p>{growth.split()[0] if growth else 'Strong'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show execution time if available
    if analysis.execution_time:
        st.caption(f"Analysis completed in {analysis.execution_time:.1f} seconds")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏢 Company Profile",
        "📊 Market Analysis",
        "🎯 Strategic Insights",
        "📈 Visualizations",
        "📋 Action Plan",
        "🔄 Changes" if analysis.diff else "ℹ️ Metadata"
    ])
    
    with tab1:
        display_company_profile(analysis)
    
    with tab2:
        display_market_analysis(analysis)
    
    with tab3:
        display_strategic_insights(analysis)
    
    with tab4:
        display_visualizations(analysis)
    
    with tab5:
        display_action_plan(analysis)
    
    with tab6:
        if analysis.diff:
            display_changes(analysis)
        else:
            display_metadata(analysis)

def display_company_profile(analysis: CompetitorAnalysis):
    """Display company profile information"""
    company_info = analysis.company_info
    company_overview = company_info.get('company_overview', {})
    product_info = company_info.get('product_info', {})
    financial_data = company_info.get('financial_data', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏢 Company Overview")
        for field in ['founded', 'employees', 'headquarters', 'industry']:
            value = company_overview.get(field, 'N/A')
            st.write(f"**{field.title()}:** {value}")
        
        description = company_overview.get('description', 'No description available')
        st.markdown("**Description:**")
        st.write(description)
        
        if financial_data:
            st.markdown("### 💰 Financial Overview")
            for field in ['total_funding', 'estimated_revenue', 'growth_rate']:
                value = financial_data.get(field, 'N/A')
                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
    
    with col2:
        st.markdown("### 🚀 Product Information")
        st.write(f"**Main Product:** {product_info.get('main_product', 'N/A')}")
        st.write(f"**Pricing Model:** {product_info.get('pricing_model', 'N/A')}")
        
        target_markets = product_info.get('target_market', [])
        if target_markets:
            st.write(f"**Target Markets:** {', '.join(target_markets)}")
        
        features = product_info.get('key_features', [])
        if features:
            st.markdown("**Key Features:**")
            for feature in features[:10]:  # Limit to 10 features
                st.write(f"• {feature}")
        
        # Recent news
        news = company_info.get('recent_news', [])
        if news:
            st.markdown("### 📢 Recent News")
            for article in news[:3]:
                if isinstance(article, dict):
                    st.write(f"**{article.get('date', 'N/A')}** - {article.get('title', 'N/A')}")
                    if article.get('source'):
                        st.caption(f"Source: {article['source']}")

def display_market_analysis(analysis: CompetitorAnalysis):
    """Display market analysis including SWOT"""
    market_analysis = analysis.market_analysis
    positioning = market_analysis.get('market_positioning', {})
    swot = market_analysis.get('swot_analysis', {})
    competitive_landscape = market_analysis.get('competitive_landscape', {})
    
    # Market positioning
    st.markdown("### 🎯 Market Positioning")
    col1, col2 = st.columns(2)
    
    with col1:
        for field in ['segment', 'position', 'market_share', 'growth_trajectory']:
            value = positioning.get(field, 'N/A')
            st.write(f"**{field.replace('_', ' ').title()}:** {value}")
    
    with col2:
        advantages = positioning.get('competitive_advantages', [])
        if advantages:
            st.markdown("**Competitive Advantages:**")
            for advantage in advantages:
                st.write(f"• {advantage}")
    
    st.markdown("---")
    
    # SWOT Analysis
    st.markdown("### 📊 SWOT Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Strengths
        strengths = swot.get('strengths', [])
        if strengths:
            st.markdown("#### 💪 Strengths")
            for item in strengths[:5]:
                st.write(f"• {item}")
        
        # Opportunities
        opportunities = swot.get('opportunities', [])
        if opportunities:
            st.markdown("#### 💡 Opportunities")
            for item in opportunities[:5]:
                st.write(f"• {item}")
    
    with col2:
        # Weaknesses
        weaknesses = swot.get('weaknesses', [])
        if weaknesses:
            st.markdown("#### ⚠️ Weaknesses")
            for item in weaknesses[:5]:
                st.write(f"• {item}")
        
        # Threats
        threats = swot.get('threats', [])
        if threats:
            st.markdown("#### 🚨 Threats")
            for item in threats[:5]:
                st.write(f"• {item}")
    
    # Competitive landscape
    if competitive_landscape:
        st.markdown("---")
        st.markdown("### 🏟️ Competitive Landscape")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for field in ['direct_competitors', 'competitive_moats']:
                items = competitive_landscape.get(field, [])
                if items:
                    st.markdown(f"**{field.replace('_', ' ').title()}:**")
                    for item in items[:5]:
                        st.write(f"• {item}")
        
        with col2:
            for field in ['indirect_competitors', 'vulnerability_areas']:
                items = competitive_landscape.get(field, [])
                if items:
                    st.markdown(f"**{field.replace('_', ' ').title()}:**")
                    for item in items[:5]:
                        st.write(f"• {item}")

def display_strategic_insights(analysis: CompetitorAnalysis):
    """Display strategic recommendations"""
    insights = analysis.strategic_insights or []
    
    st.markdown("### 🎯 Strategic Recommendations")
    
    if not insights:
        st.info("No strategic recommendations available.")
        return
    
    for i, insight in enumerate(insights):
        if isinstance(insight, dict):
            category = insight.get('category', f'Strategy {i+1}')
            priority = insight.get('priority', 'Medium')
            timeline = insight.get('timeline', 'TBD')
            
            priority_color = {
                'Critical': '🔴',
                'High': '🟠',
                'Medium': '🟡',
                'Low': '🟢'
            }.get(priority, '⚪')
            
            with st.expander(f"{priority_color} {category} - Priority: {priority}"):
                st.write(f"**Timeline:** {timeline}")
                st.write(f"**Expected Impact:** {insight.get('expected_impact', 'Moderate improvement')}")
                
                recommendations = insight.get('recommendations', [])
                if recommendations:
                    st.markdown("**Recommendations:**")
                    for j, rec in enumerate(recommendations, 1):
                        st.write(f"{j}. {rec}")
        else:
            st.write(f"• {insight}")
    
    # Threat assessment
    threat_assessment = analysis.market_analysis.get('threat_assessment', {})
    
    if threat_assessment:
        st.markdown("---")
        st.markdown("### ⚠️ Threat Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            threat_level = threat_assessment.get('overall_threat_level', 'Medium')
            st.write(f"**Overall Threat Level:** {threat_level}")
            
            threat_factors = threat_assessment.get('key_threat_factors', [])
            if threat_factors:
                st.markdown("**Key Threat Factors:**")
                for factor in threat_factors:
                    st.write(f"• {factor}")
        
        with col2:
            mitigation = threat_assessment.get('mitigation_strategies', [])
            if mitigation:
                st.markdown("**Mitigation Strategies:**")
                for strategy in mitigation:
                    st.write(f"• {strategy}")

def display_visualizations(analysis: CompetitorAnalysis):
    """Display interactive visualizations"""
    st.markdown("### 📊 Competitive Analysis Visualizations")
    
    # Create sample data for visualizations
    competitor_name = analysis.competitor_name
    
    # 1. Competitive Positioning Matrix
    st.markdown("#### Market Position vs Innovation Score")
    
    positioning_data = pd.DataFrame({
        'Company': [competitor_name, 'Competitor A', 'Competitor B', 'Your Company'],
        'Market_Share': [12, 25, 18, 8],
        'Innovation_Score': [85, 70, 75, 80],
        'Customer_Satisfaction': [4.3, 4.0, 4.2, 4.1],
        'Financial_Health': [90, 85, 80, 75]
    })
    
    fig1 = px.scatter(
        positioning_data,
        x='Market_Share',
        y='Innovation_Score',
        size='Customer_Satisfaction',
        color='Company',
        title='Competitive Positioning Matrix',
        hover_data=['Financial_Health'],
        size_max=60
    )
    
    fig1.update_layout(
        xaxis_title="Market Share (%)",
        yaxis_title="Innovation Score",
        height=500
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Competitive Strengths Radar
    st.markdown("#### Competitive Strengths Analysis")
    
    categories = ['Technology', 'Market Position', 'Financial Health',
                  'Product Quality', 'Customer Base', 'Brand Recognition']
    
    fig2 = go.Figure()
    
    # Add competitor trace
    fig2.add_trace(go.Scatterpolar(
        r=[8, 7, 9, 8, 6, 7],
        theta=categories,
        fill='toself',
        name=competitor_name,
        line_color='#ff7f0e'
    ))
    
    # Add your company trace
    fig2.add_trace(go.Scatterpolar(
        r=[7, 5, 6, 8, 8, 6],
        theta=categories,
        fill='toself',
        name='Your Company',
        line_color='#1f77b4'
    ))
    
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        title="Competitive Strengths Comparison",
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Growth Trends
    st.markdown("#### Market Growth Trends")
    
    # Generate trend data
    dates = pd.date_range(start='2023-01-01', periods=8, freq='Q')
    
    trends_data = pd.DataFrame({
        'Date': dates,
        f'{competitor_name}_Growth': [50, 65, 85, 115, 155, 200, 250, 310],
        'Your_Company_Growth': [30, 42, 58, 78, 105, 140, 185, 240],
        'Industry_Average': [80, 88, 98, 110, 125, 142, 162, 185]
    })
    
    fig3 = px.line(
        trends_data,
        x='Date',
        y=[f'{competitor_name}_Growth', 'Your_Company_Growth', 'Industry_Average'],
        title='Growth Trajectory Comparison (Indexed to 100)',
        labels={'value': 'Growth Index', 'variable': 'Company'}
    )
    
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Feature Comparison
    st.markdown("#### Product Feature Comparison")
    
    features_data = pd.DataFrame({
        'Feature': ['AI Integration', 'Mobile App', 'API Access',
                    'Analytics', 'Security', 'User Experience'],
        competitor_name: [9, 8, 7, 9, 8, 9],
        'Industry Average': [6, 7, 8, 7, 9, 7],
        'Your Company': [7, 9, 6, 8, 9, 8]
    })
    
    fig4 = px.bar(
        features_data,
        x='Feature',
        y=[competitor_name, 'Industry Average', 'Your Company'],
        title='Feature Strength Comparison (1-10 Scale)',
        barmode='group'
    )
    
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

def display_action_plan(analysis: CompetitorAnalysis):
    """Display strategic action plan and KPIs"""
    st.markdown("### 🎯 Strategic Action Plan")
    
    # Timeline-based actions
    action_categories = [
        {
            "title": "Immediate Actions (0-30 Days)",
            "emoji": "🚀",
            "actions": [
                "Set up competitive monitoring dashboard with real-time alerts",
                "Conduct feature-by-feature comparison analysis",
                "Survey customers about competitor awareness",
                "Review and optimize pricing strategy",
                "Identify strategic partnership opportunities",
                "Establish competitive intelligence briefings"
            ]
        },
        {
            "title": "Short-term Strategy (1-6 Months)",
            "emoji": "📈",
            "actions": [
                "Develop differentiating features based on gaps",
                "Launch targeted marketing campaign",
                "Strengthen customer retention programs",
                "Explore new market segments",
                "Enhance product integrations",
                "Recruit key talent from competitors"
            ]
        },
        {
            "title": "Long-term Vision (6+ Months)",
            "emoji": "🔮",
            "actions": [
                "Execute international expansion strategy",
                "Invest in AI and automation capabilities",
                "Develop enterprise sales strategy",
                "Build strategic partnerships",
                "Consider acquisition opportunities",
                "Establish thought leadership"
            ]
        }
    ]
    
    for category in action_categories:
        st.markdown(f"#### {category['emoji']} {category['title']}")
        for i, action in enumerate(category['actions'], 1):
            st.write(f"**{i}.** {action}")
        st.markdown("---")
    
    # KPI Dashboard
    st.markdown("### 📊 Key Performance Indicators (KPIs)")
    st.markdown("Monitor these metrics to track competitive performance:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    kpis = [
        ("Competitive Win Rate", "72%", "5%", "Win rate against this competitor"),
        ("Market Share Growth", "15%", "3%", "YoY market share growth"),
        ("Customer Retention", "94%", "2%", "Retention vs competitor switching"),
        ("Feature Parity", "88%", "7%", "Features matched or exceeded")
    ]
    
    for (label, value, delta, help_text), col in zip(kpis, [col1, col2, col3, col4]):
        with col:
            st.metric(
                label=label,
                value=value,
                delta=delta,
                delta_color="normal",
                help=help_text
            )
    
    # Additional metrics
    st.markdown("#### Additional Metrics to Track")
    
    additional_kpis = [
        "**Competitive Mention Share:** Industry discussion percentage",
        "**CAC Ratio:** Your CAC vs estimated competitor CAC",
        "**Release Velocity:** Feature releases per quarter",
        "**Brand Sentiment:** Net sentiment vs competitor",
        "**Sales Cycle:** Average cycle when competing directly",
        "**CLV Premium:** Customer lifetime value advantage"
    ]
    
    for kpi in additional_kpis:
        st.write(f"• {kpi}")

def display_changes(analysis: CompetitorAnalysis):
    """Display changes from previous analysis"""
    st.markdown("### 🔄 Changes from Previous Analysis")
    
    if not analysis.diff:
        st.info("No previous analysis available for comparison.")
        return
    
    diff = analysis.diff
    
    # Summary changes
    if diff.get('summary'):
        st.markdown("#### 📊 Key Changes")
        for key, change in diff['summary'].items():
            if isinstance(change, dict):
                from_val = change.get('from', 'N/A')
                to_val = change.get('to', 'N/A')
                delta = change.get('change')
                
                if delta and isinstance(delta, (int, float)):
                    st.metric(
                        label=key.replace('_', ' ').title(),
                        value=to_val,
                        delta=f"{delta:+.2f}" if isinstance(delta, float) else f"{delta:+d}"
                    )
                else:
                    st.write(f"**{key.replace('_', ' ').title()}:** {from_val} → {to_val}")
    
    # Detailed changes
    if diff.get('details'):
        st.markdown("#### 📝 Detailed Changes")
        for category, changes in diff['details'].items():
            st.write(f"**{category.replace('_', ' ').title()}:**")
            if isinstance(changes, dict):
                for key, value in changes.items():
                    st.write(f"  • {key}: {value}")
            else:
                st.write(f"  {changes}")
    
    # New items
    if diff.get('new_items'):
        st.markdown("#### ✨ New Items")
        for category, items in diff['new_items'].items():
            st.write(f"**{category.replace('_', ' ').title()}:**")
            for item in items[:5]:  # Limit display
                st.write(f"  • {item}")

def display_metadata(analysis: CompetitorAnalysis):
    """Display analysis metadata"""
    st.markdown("### ℹ️ Analysis Metadata")
    
    metadata = {
        "Generated At": analysis.generated_at,
        "Status": analysis.status or "Complete",
        "Confidence Score": f"{analysis.confidence_score:.1%}",
        "Execution Time": f"{analysis.execution_time:.1f}s" if analysis.execution_time else "N/A"
    }
    
    for key, value in metadata.items():
        st.write(f"**{key}:** {value}")
    
    if analysis.warnings:
        st.markdown("#### ⚠️ Analysis Warnings")
        for warning in analysis.warnings:
            st.warning(warning)

# -----------------------------
# Main Application
# -----------------------------

def run_competitive_analysis(competitor_name: str, config: Dict[str, Any]) -> Optional[CompetitorAnalysis]:
    """Execute competitive analysis with proper orchestration"""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Check if agents are available
        if not AGENTS_OK:
            st.error("Required agent modules are not available. Please check installation.")
            return None
        
        # Initialize agents
        data_agent = create_data_collection_agent(
            demo_mode=config.get('demo_mode', False),
            serpapi_key=config.get('serpapi_key')
        )
        
        analysis_engine = AIAnalysisEngine(
            openai_client=config.get('openai_client'),
            anthropic_client=config.get('anthropic_client')
        )
        
        # Configure orchestrator
        orch_config = OrchestratorConfig(
            max_plan_loops=config.get('max_iterations', 2),
            confidence_threshold=config.get('confidence_threshold', 0.7),
            timeout_collect_sec=float(config.get('timeout_seconds', 30)),
            timeout_analyze_sec=float(config.get('timeout_seconds', 40)),
            simulated_delays=config.get('demo_mode', False),
            enable_retry_on_timeout=True,
            attach_diff=True
        )
        
        # Create UI callbacks
        ui_callbacks = StreamlitUICallbacks(progress_bar, status_text)
        
        # Initialize orchestrator
        orchestrator = CompeteAIOrchestrator(
            data_agent=data_agent,
            analysis_engine=analysis_engine,
            ui_callbacks=ui_callbacks,
            config=orch_config
        )
        
        # Prepare user context
        user_context = {
            'company_name': config.get('company_name'),
            'company_focus': config.get('company_focus'),
            'industry': config.get('industry'),
            'company_stage': config.get('company_stage'),
            'team_size': config.get('team_size'),
            'analysis_depth': config.get('analysis_depth'),
            'include_financials': config.get('include_financials')
        }
        
        # Run analysis with previous snapshot for delta tracking
        result = orchestrator.run_analysis(
            competitor_name=competitor_name,
            user_context=user_context,
            previous_snapshot=st.session_state.last_result_dict
        )
        
        # Convert to CompetitorAnalysis dataclass
        analysis = CompetitorAnalysis(
            competitor_name=result.get('competitor_name', competitor_name),
            company_info=result.get('company_info', {}),
            market_analysis=result.get('market_analysis', {}),
            strategic_insights=result.get('strategic_insights', []),
            confidence_score=float(result.get('confidence_score', 0.75)),
            generated_at=result.get('generated_at', datetime.now().isoformat()),
            status=result.get('status', 'ok'),
            warnings=result.get('warnings'),
            diff=result.get('diff'),
            execution_time=result.get('execution_time_sec')
        )
        
        # Store raw result for next diff
        st.session_state.last_result_dict = result
        
        return analysis
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        st.error(f"❌ Analysis failed: {str(e)}")
        return None
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Setup sidebar configuration
    config = setup_sidebar()
    
    # Main header
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(LOGO_PATH, width=80)
    with col_title:
        st.markdown('<h1 class="main-header">CompeteAI Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-Powered Competitive Intelligence & Strategic Analysis")
    
    # Demo mode notification
    if config['demo_mode']:
        st.markdown("""
        <div class="notification success-notification">
            <h3>🚀 Demo Mode Active</h3>
            <p><strong>Welcome to CompeteAI!</strong> Currently running in demo mode with realistic mock data.</p>
            <p><strong>Features:</strong> Market positioning • SWOT analysis • Strategic recommendations • 
            Threat assessment • Interactive visualizations • Actionable insights</p>
            <p><em>Add API keys in sidebar to unlock live AI analysis.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main analysis interface
    st.markdown("---")
    st.markdown("## 🔍 Competitive Analysis")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        competitor_name = st.text_input(
            "Enter competitor name for analysis:",
            placeholder="e.g., Notion, Slack, Airtable, Monday.com, Asana, Linear",
            help="Enter any company name to perform comprehensive competitive intelligence analysis"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button(
            "🚀 Analyze",
            type="primary",
            use_container_width=True
        )
    
    with col3:
        st.write("")
        st.write("")
        if st.session_state.analysis_history:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.analysis_results = None
                st.session_state.last_result_dict = None
                st.rerun()
    
    # Execute analysis
    if analyze_button and competitor_name:
        if not competitor_name.strip():
            st.error("Please enter a competitor name to analyze.")
        else:
            with st.spinner("Initializing AI analysis engines..."):
                # Run analysis
                analysis = run_competitive_analysis(competitor_name.strip(), config)
                
                if analysis:
                    # Store results
                    st.session_state.analysis_results = analysis
                    st.session_state.analysis_history.append({
                        'competitor': competitor_name.strip(),
                        'timestamp': datetime.now().isoformat(),
                        'confidence': f"{analysis.confidence_score:.0%}",
                        'status': analysis.status
                    })
                    
                    st.success(f"✅ Analysis complete for {competitor_name}!")
    
    # Display results or landing page
    if st.session_state.analysis_results:
        st.markdown("---")
        display_analysis_results(st.session_state.analysis_results)
    else:
        display_landing_page()
    
    # Display history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### 📚 Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(history_df, use_container_width=True)
    
    # Footer
    display_footer()

def display_landing_page():
    """Display landing page when no analysis is active"""
    st.markdown("---")
    st.markdown("## 🚀 Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Intelligence Gathering
        - **Automated Data Collection**
        - **Financial Analysis**
        - **Product Intelligence**
        - **Market Monitoring**
        - **Social Sentiment**
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 AI Analysis Engine
        - **Multi-Agent Architecture**
        - **Cross-Validation**
        - **SWOT Analysis**
        - **Market Positioning**
        - **Predictive Insights**
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Strategic Intelligence
        - **Actionable Recommendations**
        - **Threat Assessment**
        - **Opportunity Identification**
        - **Performance Visualization**
        - **KPI Tracking**
        """)
    
    st.markdown("---")
    st.info("💡 **Get Started:** Enter a competitor name above and click 'Analyze' to begin!")

def display_footer():
    """Display application footer"""
    st.markdown("---")
    
    # Create two columns for logo and text
    col1, col2 = st.columns([1, 5])
    
    # Display logo in the first column
    with col1:
        st.image(LOGO_PATH, width=100)
    
    # Display footer text in the second column
    with col2:
        footer_html = f"""
        <div style='color: #666; padding: 20px 10px;'>
            <h3 style='color: #1f77b4; margin-bottom: 15px;'>CompeteAI Platform</h3>
            <p style='font-size: 1.1em; margin-bottom: 10px;'>
                <strong>Advanced Competitive Intelligence & Strategic Analysis</strong>
            </p>
            <p>Multi-Agent AI Architecture | Real-time Market Analysis | Strategic Insights | Actionable Recommendations</p>
            <p style='margin-top: 15px;'><em>Powered by OpenAI GPT-4 and Anthropic Claude | Built with Streamlit</em></p>
            <p style='margin-top: 10px; font-size: 0.9em;'>
                Transform your competitive strategy with AI-powered intelligence
            </p>
            <p style='margin-top: 8px; font-size: 0.85em; color: #999;'>
                Version: {app_version}
            </p>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)

# -----------------------------
# Application Entry Point
# -----------------------------

if __name__ == "__main__":
    main()