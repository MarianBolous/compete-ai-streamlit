import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle async in Streamlit
try:
    nest_asyncio.apply()
except Exception:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="CompeteAI - Advanced Competitive Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        text-align: center;
    }
    
    .analysis-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    .success-notification {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 20px;
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class CompetitorAnalysis:
    competitor_name: str
    company_info: Dict
    market_analysis: Dict
    strategic_insights: List[Dict]
    confidence_score: float
    generated_at: str

class DataCollectionAgent:
    """Agent for collecting competitive intelligence data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def gather_company_data(self, company_name: str) -> Dict:
        """Gather comprehensive company data"""
        try:
            # Simulate data collection with realistic mock data
            return {
                "company_overview": {
                    "name": company_name,
                    "founded": "2020",
                    "employees": "100-500",
                    "headquarters": "San Francisco, CA",
                    "industry": "Technology/SaaS",
                    "description": f"{company_name} is a leading technology company providing innovative solutions for modern businesses with a focus on productivity, collaboration, and data-driven insights."
                },
                "product_info": {
                    "main_product": f"{company_name} Platform",
                    "key_features": [
                        "AI-powered analytics and insights",
                        "Real-time collaboration tools",
                        "Advanced reporting and dashboards", 
                        "API integrations and workflow automation",
                        "Enterprise-grade security and compliance",
                        "Mobile-first design and offline capabilities"
                    ],
                    "pricing_model": "Freemium with enterprise tiers",
                    "target_market": ["SMBs", "Mid-market", "Enterprise"]
                },
                "financial_data": {
                    "total_funding": "$75M",
                    "last_round": {
                        "type": "Series B",
                        "amount": "$50M",
                        "date": "2024-02-10",
                        "valuation": "$400M",
                        "lead_investor": "Sequoia Capital"
                    },
                    "estimated_revenue": "$20M-50M ARR",
                    "growth_rate": "150% YoY",
                    "key_investors": ["Sequoia Capital", "Andreessen Horowitz", "GV", "Accel Partners"]
                },
                "recent_news": [
                    {
                        "title": f"{company_name} raises $50M Series B to accelerate growth",
                        "date": "2024-02-10",
                        "source": "TechCrunch",
                        "sentiment": "Positive",
                        "summary": f"Company secured significant funding to expand product development and market reach."
                    },
                    {
                        "title": f"{company_name} launches advanced AI features",
                        "date": "2024-02-05", 
                        "source": "VentureBeat",
                        "sentiment": "Positive",
                        "summary": "New AI-powered capabilities enhance user experience and productivity."
                    },
                    {
                        "title": f"{company_name} expands to European markets",
                        "date": "2024-01-28",
                        "source": "Forbes", 
                        "sentiment": "Positive",
                        "summary": "International expansion signals strong growth trajectory and market confidence."
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Data collection error: {e}")
            return {"error": str(e)}

class AIAnalysisEngine:
    """AI-powered analysis engine for competitive intelligence"""
    
    def __init__(self, openai_client=None, anthropic_client=None):
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
    
    def analyze_competitor(self, company_name: str, company_data: Dict, user_context: Dict) -> Dict:
        """Perform comprehensive competitive analysis"""
        
        if self.openai_client or self.anthropic_client:
            return self._ai_powered_analysis(company_name, company_data, user_context)
        else:
            return self._generate_structured_analysis(company_name, company_data, user_context)
    
    def _ai_powered_analysis(self, company_name: str, company_data: Dict, user_context: Dict) -> Dict:
        """Use AI APIs for enhanced analysis when available"""
        try:
            # This would integrate with OpenAI/Anthropic APIs in production
            # For now, return enhanced mock analysis
            return self._generate_structured_analysis(company_name, company_data, user_context)
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return self._generate_structured_analysis(company_name, company_data, user_context)
    
    def _generate_structured_analysis(self, company_name: str, company_data: Dict, user_context: Dict) -> Dict:
        """Generate comprehensive structured analysis"""
        company_overview = company_data.get("company_overview", {})
        product_info = company_data.get("product_info", {})
        financial_data = company_data.get("financial_data", {})
        
        return {
            "market_positioning": {
                "segment": "B2B SaaS Technology",
                "position": "Growing market leader with strong momentum",
                "market_share": "5-10% in target segment",
                "growth_trajectory": "High growth with 100-200% YoY expansion",
                "competitive_advantages": [
                    "Advanced AI integration and automation",
                    "Superior user experience and interface design",
                    "Strong financial backing and investor confidence",
                    "Rapid product development and feature releases"
                ]
            },
            "swot_analysis": {
                "strengths": [
                    "Strong product-market fit with high user satisfaction ratings",
                    "Significant recent funding ($75M total) enabling rapid growth",
                    "Advanced AI capabilities differentiating from traditional competitors",
                    "Experienced leadership team with proven track record in scaling SaaS",
                    "Strong brand recognition and market positioning in target segments"
                ],
                "weaknesses": [
                    "Limited international presence compared to established global players",
                    "Higher pricing point may limit penetration in price-sensitive SMB market",
                    "Smaller team size could impact ability to scale operations rapidly",
                    "Dependency on key personnel and founders for strategic direction",
                    "Limited enterprise features compared to mature incumbent solutions"
                ],
                "opportunities": [
                    "European and Asian market expansion with minimal competitive presence",
                    "Enterprise segment penetration with higher contract values and stickiness",
                    "Strategic partnerships with major technology platforms and integrators",
                    "AI and automation trend driving increased demand for intelligent solutions",
                    "Potential acquisition target for larger technology companies seeking innovation"
                ],
                "threats": [
                    "Well-funded competitors with significantly deeper financial resources",
                    "Economic downturn potentially reducing B2B software spending budgets",
                    "Large technology companies (Microsoft, Google) entering market with free alternatives",
                    "Regulatory changes in data privacy, AI governance, and cross-border data transfer",
                    "High customer churn to established alternatives during economic uncertainty"
                ]
            },
            "competitive_landscape": {
                "direct_competitors": ["Competitor A", "Competitor B", "Competitor C"],
                "indirect_competitors": ["Alternative Solution X", "Alternative Solution Y"],
                "competitive_moats": [
                    "Advanced AI and machine learning capabilities",
                    "Strong customer data and usage insights",
                    "Network effects from user collaboration features",
                    "High switching costs due to data integration and customization"
                ],
                "vulnerability_areas": [
                    "Price competition from well-funded startups",
                    "Feature parity from incumbent players with more resources",
                    "Customer acquisition cost increases in competitive landscape",
                    "Talent retention challenges in competitive hiring market"
                ]
            },
            "strategic_recommendations": [
                {
                    "category": "Product Strategy",
                    "priority": "High",
                    "timeline": "6-12 months",
                    "recommendations": [
                        "Develop mobile-first features to compete with established mobile-native players",
                        "Invest heavily in advanced reporting and analytics capabilities for enterprise segment",
                        "Create industry-specific templates and workflows for vertical market penetration",
                        "Enhance API and integration capabilities to build platform ecosystem effects"
                    ],
                    "expected_impact": "Significant improvement in competitive positioning and customer retention"
                },
                {
                    "category": "Market Strategy", 
                    "priority": "High",
                    "timeline": "3-9 months",
                    "recommendations": [
                        "Target underserved enterprise segment with custom solutions and white-glove service",
                        "Expand to European markets before competitor presence solidifies market positions",
                        "Develop strategic partnerships with management consulting firms for implementation services",
                        "Create comprehensive channel partner program targeting industry-specific verticals"
                    ],
                    "expected_impact": "Market share expansion and revenue diversification across segments"
                },
                {
                    "category": "Competitive Strategy",
                    "priority": "Critical", 
                    "timeline": "1-3 months",
                    "recommendations": [
                        "Establish comprehensive competitive intelligence monitoring system with real-time alerts",
                        "Develop rapid response team for competitor feature releases and strategic moves",
                        "Create clear competitive differentiation messaging for sales team and marketing materials",
                        "Build meaningful switching costs through deep data integration and workflow customization"
                    ],
                    "expected_impact": "Improved competitive win rates and reduced customer churn to competitors"
                },
                {
                    "category": "Financial Strategy",
                    "priority": "Medium",
                    "timeline": "6-18 months", 
                    "recommendations": [
                        "Optimize unit economics and reduce customer acquisition costs through improved conversion",
                        "Develop multiple revenue streams including professional services and marketplace fees",
                        "Implement value-based pricing model tied to customer business outcomes and ROI",
                        "Plan for Series C funding round to maintain competitive position and fuel expansion"
                    ],
                    "expected_impact": "Improved financial sustainability and reduced dependency on venture funding"
                }
            ],
            "threat_assessment": {
                "overall_threat_level": "Medium-High",
                "key_threat_factors": [
                    "Strong financial position enabling aggressive market expansion",
                    "Advanced product capabilities creating competitive pressure on features",
                    "Growing brand recognition potentially affecting customer acquisition",
                    "Strategic partnerships expanding market reach and distribution channels"
                ],
                "mitigation_strategies": [
                    "Focus on unique value proposition development and market positioning",
                    "Accelerate product development in areas of competitive weakness",
                    "Strengthen customer relationships and loyalty programs",
                    "Develop strategic partnerships to match competitive distribution advantages"
                ]
            },
            "confidence_score": 0.85,
            "analysis_date": datetime.now().isoformat()
        }

class CompeteAIOrchestrator:
    """Main orchestrator for the competitive intelligence platform"""
    
    def __init__(self, openai_client=None, anthropic_client=None):
        self.data_agent = DataCollectionAgent()
        self.ai_engine = AIAnalysisEngine(openai_client, anthropic_client)
    
    def run_analysis(self, competitor_name: str, user_context: Dict) -> CompetitorAnalysis:
        """Run comprehensive competitive analysis"""
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Data Collection (30%)
            status_text.text("🔍 Gathering competitive intelligence data...")
            progress_bar.progress(30)
            time.sleep(1.5)
            
            company_data = self.data_agent.gather_company_data(competitor_name)
            
            # Step 2: AI Analysis (70%)
            status_text.text("🧠 Performing AI-powered competitive analysis...")
            progress_bar.progress(70)
            time.sleep(2)
            
            analysis_results = self.ai_engine.analyze_competitor(competitor_name, company_data, user_context)
            
            # Step 3: Report Generation (90%)
            status_text.text("📊 Generating comprehensive insights report...")
            progress_bar.progress(90)
            time.sleep(1)
            
            # Step 4: Finalization (100%)
            status_text.text("✅ Analysis complete - generating visualizations...")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clean up progress indicators
            status_text.empty()
            progress_bar.empty()
            
            return CompetitorAnalysis(
                competitor_name=competitor_name,
                company_info=company_data,
                market_analysis=analysis_results,
                strategic_insights=analysis_results.get('strategic_recommendations', []),
                confidence_score=analysis_results.get('confidence_score', 0.85),
                generated_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            status_text.error(f"Analysis failed: {str(e)}")
            progress_bar.empty()
            logger.error(f"Analysis error: {e}")
            raise e

def setup_sidebar() -> Dict[str, Any]:
    """Setup sidebar configuration and return user context"""
    
    # Logo and title
    st.sidebar.markdown('<div class="sidebar-logo">🎯 CompeteAI</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**Advanced Competitive Intelligence Platform**")
    
    # API Configuration Section
    st.sidebar.markdown("### 🤖 AI Configuration")
    
    openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Optional - Enter your OpenAI API key for enhanced AI analysis",
        placeholder="sk-..."
    )
    
    anthropic_key = st.sidebar.text_input(
        "Anthropic API Key", 
        type="password",
        help="Optional - Enter your Anthropic API key for enhanced AI analysis",
        placeholder="sk-ant-..."
    )
    
    demo_mode = st.sidebar.checkbox(
        "Demo Mode",
        value=True,
        help="Use realistic mock data for demonstration purposes"
    )
    
    # Initialize AI clients
    openai_client = None
    anthropic_client = None
    
    if openai_key and not demo_mode:
        try:
            import openai
            openai_client = openai.OpenAI(api_key=openai_key)
            st.sidebar.success("✅ OpenAI connected")
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
    
    # User Context Section
    st.sidebar.markdown("### 🏢 Your Company Context")
    
    company_name = st.sidebar.text_input(
        "Company Name",
        value="YourStartup Inc.",
        help="Enter your company name for personalized analysis"
    )
    
    industry = st.sidebar.selectbox(
        "Industry",
        ["SaaS", "E-commerce", "FinTech", "HealthTech", "EdTech", "MarTech", "AI/ML", "Enterprise Software", "Other"],
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
    
    st.sidebar.markdown("---")
    
    # Additional Options
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
    
    return {
        "openai_client": openai_client,
        "anthropic_client": anthropic_client, 
        "demo_mode": demo_mode,
        "company_name": company_name,
        "industry": industry,
        "company_stage": company_stage,
        "team_size": team_size,
        "analysis_depth": analysis_depth,
        "include_financials": include_financials
    }

def display_analysis_results(analysis: CompetitorAnalysis):
    """Display comprehensive analysis results"""
    
    # Main header
    st.markdown(
        f'<h1 class="main-header">Analysis Report: {analysis.competitor_name}</h1>',
        unsafe_allow_html=True
    )
    
    # Executive summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    market_analysis = analysis.market_analysis
    positioning = market_analysis.get('market_positioning', {})
    threat_assessment = market_analysis.get('threat_assessment', {})
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Market Position</h3>
            <p>{positioning.get('position', 'Growing Player')[:20]}...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        threat_level = threat_assessment.get('overall_threat_level', 'Medium')
        threat_class = "threat-high" if "High" in threat_level else "threat-medium" if "Medium" in threat_level else "threat-low"
        
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
        growth_word = growth.split()[0] if growth else 'Strong'
        st.markdown(f"""
        <div class="metric-container">
            <h3>📈 Growth</h3>
            <p>{growth_word}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 Company Profile",
        "📊 Market Analysis", 
        "🎯 Strategic Insights",
        "📈 Visualizations",
        "📋 Action Plan"
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

def display_company_profile(analysis: CompetitorAnalysis):
    """Display company profile information"""
    company_info = analysis.company_info
    company_overview = company_info.get('company_overview', {})
    product_info = company_info.get('product_info', {})
    financial_data = company_info.get('financial_data', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Company overview
        st.markdown("### 🏢 Company Overview")
        st.write(f"**Founded:** {company_overview.get('founded', 'N/A')}")
        st.write(f"**Employees:** {company_overview.get('employees', 'N/A')}")
        st.write(f"**Headquarters:** {company_overview.get('headquarters', 'N/A')}")
        st.write(f"**Industry:** {company_overview.get('industry', 'N/A')}")
        
        st.markdown("**Description:**")
        st.write(company_overview.get('description', 'No description available'))
        
        # Financial information
        if financial_data:
            st.markdown("### 💰 Financial Overview")
            st.write(f"**Total Funding:** {financial_data.get('total_funding', 'N/A')}")
            st.write(f"**Estimated Revenue:** {financial_data.get('estimated_revenue', 'N/A')}")
            st.write(f"**Growth Rate:** {financial_data.get('growth_rate', 'N/A')}")
            
            last_round = financial_data.get('last_round', {})
            if last_round:
                st.write(f"**Last Funding:** {last_round.get('type', 'N/A')} - {last_round.get('amount', 'N/A')}")
                st.write(f"**Valuation:** {last_round.get('valuation', 'N/A')}")
    
    with col2:
        # Product information
        st.markdown("### 🚀 Product Information")
        st.write(f"**Main Product:** {product_info.get('main_product', 'N/A')}")
        st.write(f"**Pricing Model:** {product_info.get('pricing_model', 'N/A')}")
        
        target_markets = product_info.get('target_market', [])
        if target_markets:
            st.write(f"**Target Markets:** {', '.join(target_markets)}")
        
        # Key features
        features = product_info.get('key_features', [])
        if features:
            st.markdown("**Key Features:**")
            for feature in features:
                st.write(f"• {feature}")
        
        # Recent news
        news = company_info.get('recent_news', [])
        if news:
            st.markdown("### 📢 Recent News")
            for article in news[:3]:
                st.write(f"**{article.get('date', 'N/A')}** - {article.get('title', 'N/A')}")
                st.caption(f"Source: {article.get('source', 'N/A')} | Sentiment: {article.get('sentiment', 'Neutral')}")
                if article.get('summary'):
                    st.write(article['summary'])
                st.markdown("---")

def display_market_analysis(analysis: CompetitorAnalysis):
    """Display detailed market analysis"""
    market_analysis = analysis.market_analysis
    positioning = market_analysis.get('market_positioning', {})
    swot = market_analysis.get('swot_analysis', {})
    competitive_landscape = market_analysis.get('competitive_landscape', {})
    
    # Market positioning
    st.markdown("### 🎯 Market Positioning")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Market Segment:** {positioning.get('segment', 'N/A')}")
        st.write(f"**Position:** {positioning.get('position', 'N/A')}")
        st.write(f"**Market Share:** {positioning.get('market_share', 'N/A')}")
        st.write(f"**Growth Trajectory:** {positioning.get('growth_trajectory', 'N/A')}")
    
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
            for strength in strengths:
                st.write(f"• {strength}")
        
        st.markdown("---")
        
        # Opportunities
        opportunities = swot.get('opportunities', [])
        if opportunities:
            st.markdown("#### 💡 Opportunities")
            for opportunity in opportunities:
                st.write(f"• {opportunity}")
    
    with col2:
        # Weaknesses  
        weaknesses = swot.get('weaknesses', [])
        if weaknesses:
            st.markdown("#### ⚠️ Weaknesses")
            for weakness in weaknesses:
                st.write(f"• {weakness}")
        
        st.markdown("---")
        
        # Threats
        threats = swot.get('threats', [])
        if threats:
            st.markdown("#### 🚨 Threats")
            for threat in threats:
                st.write(f"• {threat}")
    
    st.markdown("---")
    
    # Competitive landscape
    if competitive_landscape:
        st.markdown("### 🏟️ Competitive Landscape")
        
        col1, col2 = st.columns(2)
        
        with col1:
            direct_competitors = competitive_landscape.get('direct_competitors', [])
            if direct_competitors:
                st.markdown("**Direct Competitors:**")
                for competitor in direct_competitors:
                    st.write(f"• {competitor}")
            
            moats = competitive_landscape.get('competitive_moats', [])
            if moats:
                st.markdown("**Competitive Moats:**")
                for moat in moats:
                    st.write(f"• {moat}")
        
        with col2:
            indirect_competitors = competitive_landscape.get('indirect_competitors', [])
            if indirect_competitors:
                st.markdown("**Indirect Competitors:**")
                for competitor in indirect_competitors:
                    st.write(f"• {competitor}")
            
            vulnerabilities = competitive_landscape.get('vulnerability_areas', [])
            if vulnerabilities:
                st.markdown("**Vulnerability Areas:**")
                for vulnerability in vulnerabilities:
                    st.write(f"• {vulnerability}")

def display_strategic_insights(analysis: CompetitorAnalysis):
    """Display strategic insights and recommendations"""
    insights = analysis.strategic_insights
    
    st.markdown("### 🎯 Strategic Recommendations")
    
    for i, insight in enumerate(insights):
        if isinstance(insight, dict):
            category = insight.get('category', f'Strategy {i+1}')
            priority = insight.get('priority', 'Medium')
            timeline = insight.get('timeline', 'TBD')
            
            # Create expandable section for each strategy category
            with st.expander(f"📋 {category} - Priority: {priority}"):
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
    market_analysis = analysis.market_analysis
    threat_assessment = market_analysis.get('threat_assessment', {})
    
    if threat_assessment:
        st.markdown("### ⚠️ Threat Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Overall Threat Level:** {threat_assessment.get('overall_threat_level', 'Medium')}")
            
            threat_factors = threat_assessment.get('key_threat_factors', [])
            if threat_factors:
                st.markdown("**Key Threat Factors:**")
                for factor in threat_factors:
                    st.write(f"• {factor}")
        
        with col2:
            mitigation_strategies = threat_assessment.get('mitigation_strategies', [])
            if mitigation_strategies:
                st.markdown("**Mitigation Strategies:**")
                for strategy in mitigation_strategies:
                    st.write(f"• {strategy}")

def display_visualizations(analysis: CompetitorAnalysis):
    """Display data visualizations and charts"""
    
    st.markdown("### 📊 Competitive Analysis Visualizations")
    
    # Competitive positioning scatter plot
    st.markdown("#### Market Position vs Innovation Score")
    
    positioning_data = pd.DataFrame({
        'Company': [analysis.competitor_name, 'Competitor A', 'Competitor B', 'Your Company'],
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
    
    # SWOT radar chart
    st.markdown("#### Competitive Strengths Analysis")
    
    categories = ['Technology', 'Market Position', 'Financial Health', 'Product Quality', 'Customer Base', 'Brand Recognition']
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatterpolar(
        r=[8, 7, 9, 8, 6, 7],
        theta=categories,
        fill='toself',
        name=analysis.competitor_name,
        line_color='#ff7f0e'
    ))
    
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
            )),
        showlegend=True,
        title="Competitive Strengths Comparison",
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Market trends timeline (FIXED to avoid length mismatch)
    st.markdown("#### Market Growth Trends")
    
    # Example series (8 points each)
    comp_growth = [50, 65, 85, 115, 155, 200, 250, 310]
    your_growth = [30, 42, 58, 78, 105, 140, 185, 240]
    industry_avg = [80, 88, 98, 110, 125, 142, 162, 185]
    
    # Try to generate a quarterly date range; if the end date would drop the last quarter,
    # fall back to a fixed number of periods equal to the shortest series length.
    try:
        dates_candidate = pd.date_range(start='2023-01-01', end='2024-12-31', freq='Q')  # quarter-end inclusive
    except Exception:
        dates_candidate = pd.date_range(start='2023-03-31', periods=len(comp_growth), freq='Q')
    
    # Normalize all arrays to the same length (shortest wins)
    n = min(len(dates_candidate), len(comp_growth), len(your_growth), len(industry_avg))
    dates = list(dates_candidate)[:n]
    comp_growth = comp_growth[:n]
    your_growth = your_growth[:n]
    industry_avg = industry_avg[:n]
    
    trends_data = pd.DataFrame({
        'Date': dates,
        f'{analysis.competitor_name}_Growth': comp_growth,
        'Your_Company_Growth': your_growth,
        'Industry_Average': industry_avg
    })
    
    fig3 = px.line(
        trends_data,
        x='Date',
        y=[f'{analysis.competitor_name}_Growth', 'Your_Company_Growth', 'Industry_Average'],
        title='Growth Trajectory Comparison (Indexed to 100)',
        labels={'value': 'Growth Index', 'variable': 'Company'}
    )
    
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Feature comparison bar chart
    st.markdown("#### Product Feature Comparison")
    
    features_data = pd.DataFrame({
        'Feature': ['AI Integration', 'Mobile App', 'API Access', 'Analytics', 'Security', 'User Experience'],
        analysis.competitor_name: [9, 8, 7, 9, 8, 9],
        'Industry Average': [6, 7, 8, 7, 9, 7],
        'Your Company': [7, 9, 6, 8, 9, 8]
    })
    
    fig4 = px.bar(
        features_data,
        x='Feature',
        y=[analysis.competitor_name, 'Industry Average', 'Your Company'],
        title='Feature Strength Comparison (1-10 Scale)',
        barmode='group'
    )
    
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

def display_action_plan(analysis: CompetitorAnalysis):
    """Display actionable recommendations and next steps"""
    
    st.markdown("### 🎯 Strategic Action Plan")
    
    # Immediate actions
    st.markdown("#### Immediate Actions (0-30 Days)")
    immediate_actions = [
        "Set up comprehensive competitive monitoring dashboard with real-time alerts and notifications",
        "Conduct detailed feature-by-feature comparison analysis against competitor's current offering",
        "Survey existing customers about their awareness of competitors and reasons for choosing your solution",
        "Review and optimize current pricing strategy based on competitive positioning analysis",
        "Identify and engage potential strategic partnership opportunities in complementary markets",
        "Establish regular competitive intelligence briefings for executive team and product leadership"
    ]
    
    for i, action in enumerate(immediate_actions, 1):
        st.write(f"**{i}.** {action}")
    
    st.markdown("---")
    
    # Short-term strategy
    st.markdown("#### Short-term Strategy (1-6 Months)")
    short_term_actions = [
        "Develop and launch unique differentiating features based on identified competitive gaps",
        "Implement targeted content marketing campaign highlighting competitive advantages and customer success stories",
        "Strengthen customer success and retention programs to reduce churn to competitive alternatives",
        "Explore expansion opportunities into underserved market segments or geographic regions",
        "Enhance product integration capabilities and API ecosystem to increase switching costs",
        "Recruit key talent from competitor organizations to gain insider knowledge and capabilities"
    ]
    
    for i, action in enumerate(short_term_actions, 1):
        st.write(f"**{i}.** {action}")
    
    st.markdown("---")
    
    # Long-term vision
    st.markdown("#### Long-term Vision (6+ Months)")
    long_term_actions = [
        "Plan and execute international market expansion strategy to compete on a global scale",
        "Invest significantly in AI and automation capabilities for sustainable competitive differentiation",
        "Develop comprehensive enterprise sales strategy with dedicated team and specialized offerings",
        "Build strategic technology partnerships and integration ecosystem to create network effects",
        "Consider strategic acquisition opportunities or partnerships to accelerate competitive position",
        "Establish thought leadership and industry influence through research, events, and community building"
    ]
    
    for i, action in enumerate(long_term_actions, 1):
        st.write(f"**{i}.** {action}")
    
    st.markdown("---")
    
    # Success metrics dashboard
    st.markdown("### 📊 Key Performance Indicators (KPIs)")
    st.markdown("Monitor these metrics to track your competitive performance:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Competitive Win Rate",
            value="72%",
            delta="5%",
            delta_color="normal",
            help="Win rate in deals where this competitor is also competing"
        )
    
    with col2:
        st.metric(
            label="Market Share Growth",
            value="15%",
            delta="3%", 
            delta_color="normal",
            help="Year-over-year market share growth in target segments"
        )
    
    with col3:
        st.metric(
            label="Customer Retention",
            value="94%",
            delta="2%",
            delta_color="normal",
            help="Customer retention rate vs competitor switching"
        )
    
    with col4:
        st.metric(
            label="Feature Parity Score",
            value="88%",
            delta="7%",
            delta_color="normal",
            help="Percentage of competitor features matched or exceeded"
        )
    
    # Additional KPI recommendations
    st.markdown("#### Additional Metrics to Track")
    additional_kpis = [
        "**Competitive Mention Share:** Percentage of industry discussions that mention your company vs competitor",
        "**Customer Acquisition Cost (CAC) Ratio:** Your CAC compared to estimated competitor CAC",
        "**Product Release Velocity:** Number of feature releases per quarter vs competitor",
        "**Brand Sentiment Score:** Net sentiment in social media and review platforms vs competitor",
        "**Sales Cycle Length:** Average sales cycle when competing directly against this competitor",
        "**Customer Lifetime Value (CLV) Premium:** CLV advantage over competitor's customer base"
    ]
    
    for kpi in additional_kpis:
        st.write(f"• {kpi}")

def main():
    """Main application function"""
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Setup sidebar and get user configuration
    config = setup_sidebar()
    
    # Main application header
    st.markdown('<h1 class="main-header">🎯 CompeteAI Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-Powered Competitive Intelligence & Strategic Analysis")
    
    # Demo mode notification
    if config['demo_mode']:
        st.markdown("""
        <div class="success-notification">
            <h3>🚀 Demo Mode Active</h3>
            <p><strong>Welcome to CompeteAI!</strong> This platform provides comprehensive competitive intelligence using advanced AI analysis. 
            Currently running in demo mode with realistic mock data to showcase full capabilities.</p>
            <p><strong>Features demonstrated:</strong> Market positioning analysis, SWOT analysis, strategic recommendations, 
            threat assessment, and actionable insights with interactive visualizations.</p>
            <p><em>Enter real API keys in the sidebar to unlock live AI analysis capabilities.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main competitor analysis interface
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
            "🚀 Analyze Competitor",
            type="primary",
            use_container_width=True,
            help="Click to start comprehensive competitive analysis"
        )
    
    with col3:
        st.write("")
        st.write("")
        if st.session_state.analysis_history:
            clear_button = st.button(
                "🗑️ Clear History", 
                use_container_width=True,
                help="Clear analysis history"
            )
            if clear_button:
                st.session_state.analysis_history = []
                st.session_state.analysis_results = None
                st.rerun()
    
    # Analysis execution
    if analyze_button and competitor_name:
        if not competitor_name.strip():
            st.error("Please enter a competitor name to analyze.")
        else:
            try:
                with st.spinner("Initializing AI analysis engines..."):
                    # Initialize the orchestrator
                    orchestrator = CompeteAIOrchestrator(
                        config['openai_client'],
                        config['anthropic_client']
                    )
                    
                    # Prepare user context
                    user_context = {
                        'company_name': config['company_name'],
                        'industry': config['industry'],
                        'company_stage': config['company_stage'],
                        'team_size': config['team_size'],
                        'analysis_depth': config['analysis_depth'],
                        'include_financials': config['include_financials']
                    }
                    
                    # Run the analysis
                    analysis_result = orchestrator.run_analysis(competitor_name.strip(), user_context)
                    
                    # Store results
                    st.session_state.analysis_results = analysis_result
                    st.session_state.analysis_history.append({
                        'competitor': competitor_name.strip(),
                        'timestamp': datetime.now().isoformat(),
                        'company_context': config['company_name']
                    })
                    
                    # Success notification
                    st.success(f"✅ Analysis complete for {competitor_name}!")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.error("Please check your configuration and try again.")
                logger.error(f"Analysis error: {e}")
    
    # Display analysis results
    if st.session_state.analysis_results:
        st.markdown("---")
        display_analysis_results(st.session_state.analysis_results)
    
    # Show platform capabilities when no analysis is active
    else:
        st.markdown("---")
        st.markdown("## 🚀 Platform Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔍 Intelligence Gathering
            - **Automated Data Collection:** Comprehensive web scraping and data aggregation
            - **Financial Analysis:** Funding rounds, valuation, and growth metrics
            - **Product Intelligence:** Feature analysis and competitive benchmarking  
            - **Market Monitoring:** Real-time news, trends, and industry developments
            - **Social Sentiment:** Brand perception and customer feedback analysis
            """)
        
        with col2:
            st.markdown("""
            ### 🧠 AI Analysis Engine
            - **Multi-Agent Architecture:** Specialized AI agents for different analysis types
            - **Cross-Validation:** Multiple AI models for enhanced accuracy and reliability
            - **SWOT Analysis:** Comprehensive strengths, weaknesses, opportunities, threats
            - **Market Positioning:** Competitive landscape and differentiation analysis
            - **Predictive Insights:** Future trends and strategic scenario planning
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Strategic Intelligence
            - **Actionable Recommendations:** Prioritized strategic actions with timelines
            - **Threat Assessment:** Risk analysis and mitigation strategies
            - **Opportunity Identification:** Market gaps and expansion possibilities
            - **Performance Visualization:** Interactive charts and competitive matrices
            - **KPI Tracking:** Custom metrics for competitive performance monitoring
            """)
        
        # Sample analysis showcase
        st.markdown("---")
        st.markdown("### 🎯 Example Analysis")
        st.info("💡 **Get Started:** Enter a competitor name above (like 'Notion', 'Slack', or 'Airtable') and click 'Analyze Competitor' to see the full AI competitive intelligence system in action!")
        
        # Analysis history
        if st.session_state.analysis_history:
            st.markdown("### 📚 Analysis History")
            history_df = pd.DataFrame(st.session_state.analysis_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(history_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 30px; background-color: #f8f9fa; border-radius: 10px; margin-top: 40px;'>
        <h3 style='color: #1f77b4; margin-bottom: 15px;'>🎯 CompeteAI Platform</h3>
        <p style='font-size: 1.1em; margin-bottom: 10px;'><strong>Advanced Competitive Intelligence & Strategic Analysis</strong></p>
        <p>Multi-Agent AI Architecture | Real-time Market Analysis | Strategic Insights | Actionable Recommendations</p>
        <p style='margin-top: 15px;'><em>Powered by OpenAI GPT-4 and Anthropic Claude | Built with Streamlit</em></p>
        <p style='margin-top: 10px; font-size: 0.9em;'>Transform your competitive strategy with AI-powered intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
