# agents/orchestrator.py
"""
CompeteAI Orchestrator (Agentic Conductor)
=========================================

Production-ready orchestrator with comprehensive guardrails:

- Plan → Delegate (Agent 1 collect, Agent 2 analyze) → Evaluate → Iterate → Finalize
- Timeouts per agent call (non-blocking fallbacks)
- Max loop iterations with evaluation policy (confidence threshold & key coverage)
- Partial results (schema-stable) if a stage fails
- Delta/diff tracking against previous snapshots
- UI-agnostic progress callbacks (works with Streamlit or any other UI)
- Structured logging with context
- Type-safe interfaces using Protocol

Usage:
------
from agents.orchestrator import CompeteAIOrchestrator, OrchestratorConfig

orchestrator = CompeteAIOrchestrator(
    data_agent=data_agent,
    analysis_engine=analysis_engine,
    ui_callbacks=UICallbacks(on_progress=..., on_done=...),
    config=OrchestratorConfig(max_plan_loops=2, confidence_threshold=0.7)
)

result = orchestrator.run_analysis(
    competitor_name="Competitor Inc",
    user_context={...},
    previous_snapshot={...}  # optional, for delta tracking
)
"""

from __future__ import annotations

import time
import logging
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Tuple, List, Protocol
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from enum import Enum

# Configure structured logging
logger = logging.getLogger(__name__)


# ---------------------------
# Enums & Constants
# ---------------------------

class AnalysisStatus(str, Enum):
    """Status indicators for analysis results."""
    OK = "ok"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"


class ThreatLevel(str, Enum):
    """Standardized threat levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ---------------------------
# Protocols & Interfaces
# ---------------------------

class UICallbacks(Protocol):
    """Type-safe interface for UI callbacks."""
    
    def on_progress(self, percent: int, message: str) -> None:
        """Update progress indicator."""
        ...
    
    def on_done(self) -> None:
        """Signal completion."""
        ...
    
    def on_error(self, error: str) -> None:
        """Handle errors (optional)."""
        ...


class DataAgent(Protocol):
    """Interface for data collection agent."""
    
    def gather_company_data(self, company_name: str) -> Dict[str, Any]:
        """Collect company data."""
        ...
    
    def refine(self, company_name: str, gaps: Dict[str, Any], seed: Dict[str, Any]) -> Dict[str, Any]:
        """Refine collection based on gaps (optional)."""
        ...


class AnalysisAgent(Protocol):
    """Interface for analysis engine."""
    
    def analyze_competitor(
        self, 
        company_name: str, 
        company_data: Dict[str, Any], 
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze competitor data."""
        ...


# ---------------------------
# Configuration
# ---------------------------

@dataclass
class OrchestratorConfig:
    """Runtime configuration with sensible defaults."""
    
    # Iteration & evaluation policy
    max_plan_loops: int = 2
    confidence_threshold: float = 0.70
    require_min_news: int = 2
    require_min_features: int = 3
    require_min_sources: int = 3
    
    # Timeouts (seconds)
    timeout_collect_sec: float = 30.0
    timeout_analyze_sec: float = 40.0
    timeout_refine_sec: float = 20.0
    
    # UX settings
    simulated_delays: bool = False
    delay_collect_sec: float = 1.2
    delay_analyze_sec: float = 1.6
    delay_finalize_sec: float = 0.6
    
    # ThreadPool configuration
    worker_threads: int = 4
    
    # Features
    attach_diff: bool = True
    enable_partial_results: bool = True
    enable_retry_on_timeout: bool = True
    
    # Quality thresholds
    min_data_quality_score: float = 0.5
    max_staleness_days: int = 90


# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class CoverageMetrics:
    """Metrics for evaluating data coverage."""
    news_count: int = 0
    features_count: int = 0
    sources_count: int = 0
    financial_data_present: bool = False
    reviews_present: bool = False
    jobs_count: int = 0
    
    @property
    def coverage_score(self) -> float:
        """Calculate overall coverage score (0-1)."""
        weights = {
            'news': 0.2,
            'features': 0.25,
            'sources': 0.15,
            'financial': 0.2,
            'reviews': 0.1,
            'jobs': 0.1
        }
        
        score = 0.0
        score += weights['news'] * min(1.0, self.news_count / 5)
        score += weights['features'] * min(1.0, self.features_count / 5)
        score += weights['sources'] * min(1.0, self.sources_count / 5)
        score += weights['financial'] * (1.0 if self.financial_data_present else 0.0)
        score += weights['reviews'] * (1.0 if self.reviews_present else 0.0)
        score += weights['jobs'] * min(1.0, self.jobs_count / 3)
        
        return round(score, 2)


@dataclass
class AnalysisResult:
    """Structured result from orchestrator."""
    competitor_name: str
    company_info: Dict[str, Any]
    market_analysis: Dict[str, Any]
    strategic_insights: List[Dict[str, Any]]
    confidence_score: float
    coverage_metrics: CoverageMetrics
    generated_at: str
    status: AnalysisStatus
    execution_time_sec: float = 0.0
    iterations_performed: int = 1
    diff: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI consumption."""
        result = asdict(self)
        result['status'] = self.status.value
        result['coverage_metrics'] = asdict(self.coverage_metrics)
        return result


# ---------------------------
# Main Orchestrator
# ---------------------------

class CompeteAIOrchestrator:
    """
    Production-ready orchestrator with comprehensive guardrails.
    Coordinates agents, enforces timeouts, handles retries, and ensures schema stability.
    """
    
    def __init__(
        self,
        data_agent: DataAgent,
        analysis_engine: AnalysisAgent,
        ui_callbacks: Optional[UICallbacks] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize orchestrator with agents and configuration.
        
        Parameters
        ----------
        data_agent : DataAgent
            Agent responsible for data collection
        analysis_engine : AnalysisAgent
            Agent responsible for analysis
        ui_callbacks : UICallbacks, optional
            UI callback handlers
        config : OrchestratorConfig, optional
            Runtime configuration
        """
        self.data_agent = data_agent
        self.analysis_engine = analysis_engine
        self.cfg = config or OrchestratorConfig()
        
        # Setup callbacks with safe defaults
        self._ui = ui_callbacks or self._create_default_callbacks()
        
        # Initialize thread pool for timeout management
        self._pool = ThreadPoolExecutor(
            max_workers=max(2, self.cfg.worker_threads),
            thread_name_prefix="orchestrator"
        )
        
        # Performance tracking
        self._start_time: Optional[float] = None
        
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources."""
        try:
            self._pool.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    
    def run_analysis(
        self,
        competitor_name: str,
        user_context: Dict[str, Any],
        *,
        previous_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full analysis pipeline with comprehensive guardrails.
        
        Parameters
        ----------
        competitor_name : str
            Name of the competitor to analyze
        user_context : dict
            User's company context for comparison
        previous_snapshot : dict, optional
            Previous analysis for delta tracking
            
        Returns
        -------
        dict
            Analysis result with schema-stable structure
        """
        self._start_time = time.time()
        logger.info(f"Starting analysis for {competitor_name}", extra={"competitor": competitor_name})
        
        try:
            # Initialize
            self._safe_progress(5, "🚀 Initializing orchestrator & agents...")
            if self.cfg.simulated_delays:
                time.sleep(0.2)
            
            # Planning phase
            self._safe_progress(15, "🧭 Planning optimal analysis strategy...")
            plan = self._create_execution_plan(competitor_name, user_context, previous_snapshot)
            if self.cfg.simulated_delays:
                time.sleep(0.3)
            
            # Data collection phase
            self._safe_progress(30, "🔍 Gathering competitive intelligence...")
            company_data, collect_metrics = self._execute_collection_phase(competitor_name, plan)
            
            # Analysis phase with iteration
            self._safe_progress(50, "🧠 Performing strategic analysis...")
            analysis_result = self._execute_analysis_phase(
                competitor_name, 
                company_data, 
                user_context, 
                collect_metrics
            )
            
            # Finalization
            self._safe_progress(90, "📊 Finalizing insights & generating report...")
            result = self._finalize_result(
                competitor_name,
                company_data,
                analysis_result,
                collect_metrics,
                previous_snapshot
            )
            
            # Complete
            self._safe_progress(100, "✅ Analysis complete")
            self._safe_done()
            
            return result.to_dict()
            
        except Exception as e:
            logger.exception(f"Critical error in orchestrator: {e}")
            # Return safe partial result
            return self._create_error_result(competitor_name, str(e)).to_dict()
    
    # ---------------------------------------------------------------------
    # Planning & Strategy
    # ---------------------------------------------------------------------
    
    def _create_execution_plan(
        self,
        competitor_name: str,
        user_context: Dict[str, Any],
        previous_snapshot: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create execution plan based on context and previous runs.
        """
        plan = {
            "competitor_name": competitor_name,
            "focus_areas": [],
            "priority_data": [],
            "max_iterations": self.cfg.max_plan_loops,
            "requires_fresh_data": True,
            "delta_mode": previous_snapshot is not None
        }
        
        # Determine focus areas based on user context
        if user_context.get("company_focus"):
            focus = user_context["company_focus"]
            if "product" in focus.lower():
                plan["focus_areas"].append("product_features")
                plan["priority_data"].append("reviews")
            if "market" in focus.lower():
                plan["focus_areas"].append("market_positioning")
                plan["priority_data"].append("news")
            if "financial" in focus.lower() or "funding" in focus.lower():
                plan["focus_areas"].append("financial_analysis")
                plan["priority_data"].append("financial_data")
        
        # Check staleness if previous snapshot exists
        if previous_snapshot:
            prev_date = previous_snapshot.get("generated_at")
            if prev_date:
                try:
                    prev_dt = datetime.fromisoformat(prev_date)
                    if (datetime.now() - prev_dt) < timedelta(days=7):
                        plan["requires_fresh_data"] = False
                        plan["max_iterations"] = 1  # Reduce iterations for recent data
                except ValueError:
                    pass
        
        logger.info(f"Execution plan created: {plan}")
        return plan
    
    # ---------------------------------------------------------------------
    # Collection Phase
    # ---------------------------------------------------------------------
    
    def _execute_collection_phase(
        self, 
        competitor_name: str, 
        plan: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], CoverageMetrics]:
        """
        Execute data collection with retries and timeout handling.
        """
        max_retries = 2 if self.cfg.enable_retry_on_timeout else 1
        last_error = None
        
        for attempt in range(max_retries):
            if attempt > 0:
                self._safe_progress(
                    35 + (attempt * 5), 
                    f"🔄 Retrying data collection (attempt {attempt + 1}/{max_retries})..."
                )
            
            if self.cfg.simulated_delays:
                time.sleep(self.cfg.delay_collect_sec)
            
            try:
                # Attempt collection with timeout
                company_data = self._collect_with_timeout(competitor_name, plan)
                
                # Validate and calculate metrics
                company_data = self._ensure_schema_stability(company_data, competitor_name)
                metrics = self._calculate_coverage_metrics(company_data)
                
                # Check if we have minimum viable data
                if metrics.coverage_score >= self.cfg.min_data_quality_score:
                    logger.info(f"Collection successful with coverage score: {metrics.coverage_score}")
                    return company_data, metrics
                
                logger.warning(f"Low coverage score: {metrics.coverage_score}, may retry")
                
            except FuturesTimeout:
                last_error = "Data collection timeout"
                logger.error(f"Collection timeout on attempt {attempt + 1}")
            except Exception as e:
                last_error = str(e)
                logger.exception(f"Collection error on attempt {attempt + 1}: {e}")
        
        # Fallback to minimal data
        logger.warning(f"Collection failed after {max_retries} attempts, using fallback")
        return self._create_fallback_company_data(competitor_name, last_error), CoverageMetrics()
    
    def _collect_with_timeout(self, competitor_name: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data collection with timeout protection.
        """
        def _collect():
            with self._log_context(agent="DataCollectionAgent", competitor=competitor_name):
                return self.data_agent.gather_company_data(competitor_name)
        
        future = self._pool.submit(_collect)
        return future.result(timeout=self.cfg.timeout_collect_sec)
    
    # ---------------------------------------------------------------------
    # Analysis Phase
    # ---------------------------------------------------------------------
    
    def _execute_analysis_phase(
        self,
        competitor_name: str,
        company_data: Dict[str, Any],
        user_context: Dict[str, Any],
        collect_metrics: CoverageMetrics
    ) -> Dict[str, Any]:
        """
        Execute analysis with iteration based on confidence and coverage.
        """
        best_analysis = {}
        best_confidence = 0.0
        iterations = 0
        
        for loop_idx in range(self.cfg.max_plan_loops):
            iterations += 1
            progress = 50 + int(30 * (loop_idx / max(1, self.cfg.max_plan_loops - 1)))
            self._safe_progress(
                progress,
                f"🧠 Analyzing competitive position (iteration {loop_idx + 1}/{self.cfg.max_plan_loops})..."
            )
            
            if self.cfg.simulated_delays:
                time.sleep(self.cfg.delay_analyze_sec)
            
            # Run analysis with timeout
            analysis = self._analyze_with_timeout(competitor_name, company_data, user_context)
            confidence = float(analysis.get("confidence_score", 0.0))
            
            # Track best result
            if confidence > best_confidence:
                best_analysis = analysis
                best_confidence = confidence
            
            # Evaluate if we should continue
            should_continue = self._should_iterate(
                confidence, 
                collect_metrics, 
                loop_idx, 
                self.cfg.max_plan_loops
            )
            
            if not should_continue:
                logger.info(f"Analysis complete after {iterations} iterations with confidence {confidence}")
                break
            
            # Try to refine data collection
            if loop_idx < self.cfg.max_plan_loops - 1:
                gaps = self._identify_gaps(company_data, collect_metrics)
                if gaps:
                    self._safe_progress(
                        progress + 5,
                        "🔄 Refining data collection based on gaps..."
                    )
                    company_data = self._refine_collection(competitor_name, company_data, gaps)
                    collect_metrics = self._calculate_coverage_metrics(company_data)
        
        # Ensure we have valid analysis
        if not best_analysis:
            best_analysis = self._create_fallback_analysis(competitor_name)
        
        best_analysis["iterations_performed"] = iterations
        return best_analysis
    
    def _analyze_with_timeout(
        self,
        competitor_name: str,
        company_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute analysis with timeout protection.
        """
        def _analyze():
            with self._log_context(agent="AIAnalysisEngine", competitor=competitor_name):
                return self.analysis_engine.analyze_competitor(
                    competitor_name,
                    company_data,
                    user_context
                )
        
        try:
            future = self._pool.submit(_analyze)
            analysis = future.result(timeout=self.cfg.timeout_analyze_sec)
            return self._ensure_analysis_schema(analysis)
        except FuturesTimeout:
            logger.error("Analysis timeout, using fallback")
            return self._create_fallback_analysis(competitor_name)
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            return self._create_fallback_analysis(competitor_name)
    
    # ---------------------------------------------------------------------
    # Evaluation & Iteration
    # ---------------------------------------------------------------------
    
    def _should_iterate(
        self,
        confidence: float,
        metrics: CoverageMetrics,
        current_iteration: int,
        max_iterations: int
    ) -> bool:
        """
        Determine if another iteration is warranted.
        """
        if current_iteration >= max_iterations - 1:
            return False
        
        # Check confidence threshold
        if confidence < self.cfg.confidence_threshold:
            logger.info(f"Confidence {confidence} below threshold {self.cfg.confidence_threshold}")
            return True
        
        # Check coverage requirements
        if metrics.news_count < self.cfg.require_min_news:
            logger.info(f"News count {metrics.news_count} below minimum {self.cfg.require_min_news}")
            return True
        
        if metrics.features_count < self.cfg.require_min_features:
            logger.info(f"Features count {metrics.features_count} below minimum {self.cfg.require_min_features}")
            return True
        
        if metrics.coverage_score < self.cfg.min_data_quality_score:
            logger.info(f"Coverage score {metrics.coverage_score} below minimum {self.cfg.min_data_quality_score}")
            return True
        
        return False
    
    def _identify_gaps(
        self,
        company_data: Dict[str, Any],
        metrics: CoverageMetrics
    ) -> Dict[str, Any]:
        """
        Identify specific data gaps for refinement.
        """
        gaps = {}
        
        if metrics.news_count < self.cfg.require_min_news:
            gaps["news"] = {
                "current": metrics.news_count,
                "required": self.cfg.require_min_news,
                "priority": "high"
            }
        
        if metrics.features_count < self.cfg.require_min_features:
            gaps["features"] = {
                "current": metrics.features_count,
                "required": self.cfg.require_min_features,
                "priority": "high"
            }
        
        if not metrics.financial_data_present:
            gaps["financial"] = {
                "current": "missing",
                "required": "basic_metrics",
                "priority": "medium"
            }
        
        if not metrics.reviews_present:
            gaps["reviews"] = {
                "current": "missing",
                "required": "platform_coverage",
                "priority": "low"
            }
        
        return gaps
    
    def _refine_collection(
        self,
        competitor_name: str,
        current_data: Dict[str, Any],
        gaps: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempt to refine data collection based on identified gaps.
        """
        try:
            # Check if agent supports refinement
            if hasattr(self.data_agent, 'refine'):
                def _refine():
                    return self.data_agent.refine(competitor_name, gaps=gaps, seed=current_data)
                
                future = self._pool.submit(_refine)
                refined_data = future.result(timeout=self.cfg.timeout_refine_sec)
                
                # Merge refined data with current data
                return self._merge_company_data(current_data, refined_data)
            else:
                # Fallback to re-collection
                logger.info("Agent doesn't support refinement, attempting re-collection")
                return self._collect_with_timeout(competitor_name, {"focus": list(gaps.keys())})
                
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return current_data
    
    # ---------------------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------------------
    
    def _finalize_result(
        self,
        competitor_name: str,
        company_data: Dict[str, Any],
        analysis: Dict[str, Any],
        metrics: CoverageMetrics,
        previous_snapshot: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Package final result with all metadata and diffs.
        """
        execution_time = time.time() - self._start_time if self._start_time else 0.0
        
        # Determine overall status
        status = self._determine_status(analysis, metrics)
        
        # Generate warnings
        warnings = self._generate_warnings(company_data, analysis, metrics)
        
        # Create result
        result = AnalysisResult(
            competitor_name=competitor_name,
            company_info=company_data,
            market_analysis=analysis,
            strategic_insights=analysis.get("strategic_recommendations", []),
            confidence_score=float(analysis.get("confidence_score", 0.5)),
            coverage_metrics=metrics,
            generated_at=datetime.now().isoformat(),
            status=status,
            execution_time_sec=round(execution_time, 2),
            iterations_performed=analysis.get("iterations_performed", 1),
            warnings=warnings,
            metadata={
                "orchestrator_version": "2.0.0",
                "config": {
                    "max_loops": self.cfg.max_plan_loops,
                    "confidence_threshold": self.cfg.confidence_threshold,
                    "timeouts": {
                        "collect": self.cfg.timeout_collect_sec,
                        "analyze": self.cfg.timeout_analyze_sec
                    }
                }
            }
        )
        
        # Compute diff if previous snapshot provided
        if self.cfg.attach_diff and previous_snapshot:
            result.diff = self._compute_comprehensive_diff(previous_snapshot, result.to_dict())
        
        return result
    
    def _determine_status(self, analysis: Dict[str, Any], metrics: CoverageMetrics) -> AnalysisStatus:
        """
        Determine overall analysis status.
        """
        confidence = float(analysis.get("confidence_score", 0.0))
        
        if confidence >= self.cfg.confidence_threshold and metrics.coverage_score >= 0.7:
            return AnalysisStatus.OK
        elif confidence >= 0.5 or metrics.coverage_score >= 0.5:
            return AnalysisStatus.PARTIAL
        else:
            return AnalysisStatus.ERROR
    
    def _generate_warnings(
        self,
        company_data: Dict[str, Any],
        analysis: Dict[str, Any],
        metrics: CoverageMetrics
    ) -> List[str]:
        """
        Generate user-friendly warnings about data quality.
        """
        warnings = []
        
        if metrics.coverage_score < 0.5:
            warnings.append("⚠️ Limited data available - results may be incomplete")
        
        if metrics.news_count == 0:
            warnings.append("📰 No recent news found - market dynamics may be outdated")
        
        if not metrics.financial_data_present:
            warnings.append("💰 Financial data unavailable - revenue estimates are approximations")
        
        confidence = float(analysis.get("confidence_score", 0.0))
        if confidence < 0.6:
            warnings.append(f"🎯 Analysis confidence is {confidence:.0%} - consider manual verification")
        
        # Check data freshness
        recent_news = company_data.get("recent_news", [])
        if recent_news:
            try:
                latest_date = max(
                    datetime.fromisoformat(n.get("date", "2000-01-01"))
                    for n in recent_news if isinstance(n, dict) and n.get("date")
                )
                days_old = (datetime.now() - latest_date).days
                if days_old > self.cfg.max_staleness_days:
                    warnings.append(f"📅 Latest data is {days_old} days old")
            except (ValueError, TypeError):
                pass
        
        return warnings
    
    # ---------------------------------------------------------------------
    # Diff & Delta Tracking
    # ---------------------------------------------------------------------
    
    def _compute_comprehensive_diff(
        self,
        prev: Dict[str, Any],
        curr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute detailed diff between snapshots.
        """
        diff = {
            "summary": {},
            "details": {},
            "metrics_changes": {},
            "new_items": {},
            "removed_items": {}
        }
        
        # Summary changes
        prev_conf = float(prev.get("confidence_score", 0.0))
        curr_conf = float(curr.get("confidence_score", 0.0))
        if abs(curr_conf - prev_conf) > 0.05:
            diff["summary"]["confidence"] = {
                "from": round(prev_conf, 2),
                "to": round(curr_conf, 2),
                "change": round(curr_conf - prev_conf, 2)
            }
        
        # Coverage changes
        prev_coverage = (prev.get("coverage_metrics", {}) or {})
        curr_coverage = (curr.get("coverage_metrics", {}) or {})
        
        for metric in ["news_count", "features_count", "sources_count"]:
            prev_val = prev_coverage.get(metric, 0)
            curr_val = curr_coverage.get(metric, 0)
            if prev_val != curr_val:
                diff["metrics_changes"][metric] = {
                    "from": prev_val,
                    "to": curr_val,
                    "change": curr_val - prev_val
                }
        
        # Threat level changes
        prev_threat = self._extract_threat_level(prev)
        curr_threat = self._extract_threat_level(curr)
        if prev_threat != curr_threat:
            diff["summary"]["threat_level"] = {
                "from": prev_threat,
                "to": curr_threat
            }
        
        # New strategic insights
        prev_insights = set(self._extract_insight_keys(prev))
        curr_insights = set(self._extract_insight_keys(curr))
        
        new_insights = curr_insights - prev_insights
        if new_insights:
            diff["new_items"]["strategic_insights"] = list(new_insights)
        
        removed_insights = prev_insights - curr_insights
        if removed_insights:
            diff["removed_items"]["strategic_insights"] = list(removed_insights)
        
        # Market position changes
        prev_position = self._extract_market_position(prev)
        curr_position = self._extract_market_position(curr)
        if prev_position != curr_position:
            diff["details"]["market_position"] = {
                "from": prev_position,
                "to": curr_position
            }
        
        return diff
    
    def _extract_threat_level(self, data: Dict[str, Any]) -> str:
        """Extract threat level from analysis data."""
        try:
            return (data.get("market_analysis", {})
                      .get("threat_assessment", {})
                      .get("overall_threat_level", "Unknown"))
        except (AttributeError, TypeError):
            return "Unknown"
    
    def _extract_insight_keys(self, data: Dict[str, Any]) -> List[str]:
        """Extract strategic insight identifiers."""
        insights = []
        try:
            for insight in data.get("strategic_insights", []):
                if isinstance(insight, dict):
                    key = f"{insight.get('category', 'Unknown')}_{insight.get('priority', 'Unknown')}"
                    insights.append(key)
        except (AttributeError, TypeError):
            pass
        return insights
    
    def _extract_market_position(self, data: Dict[str, Any]) -> str:
        """Extract market position from analysis."""
        try:
            return (data.get("market_analysis", {})
                      .get("market_positioning", {})
                      .get("position", "Unknown"))
        except (AttributeError, TypeError):
            return "Unknown"
    
    # ---------------------------------------------------------------------
    # Schema Stability & Fallbacks
    # ---------------------------------------------------------------------
    
    def _ensure_schema_stability(
        self,
        data: Dict[str, Any],
        competitor_name: str
    ) -> Dict[str, Any]:
        """
        Ensure data conforms to expected schema with safe defaults.
        """
        return {
            "company_overview": data.get("company_overview") or {
                "name": competitor_name,
                "description": "",
                "industry": "Technology",
                "founded": "",
                "headquarters": "",
                "employees": "",
                "website": ""
            },
            "product_info": data.get("product_info") or {
                "main_product": f"{competitor_name} Platform",
                "key_features": [],
                "pricing_model": "Subscription",
                "target_market": ["Enterprise", "SMB"],
                "pricing": {},
                "integrations": []
            },
            "financial_data": data.get("financial_data") or {
                "revenue": "Not disclosed",
                "funding": "Not disclosed",
                "valuation": "Not disclosed",
                "growth_rate": "Not disclosed"
            },
            "recent_news": data.get("recent_news") or [],
            "jobs": data.get("jobs") or [],
            "reviews": data.get("reviews") or {
                "average_rating": 0.0,
                "total_reviews": 0,
                "platforms": {}
            },
            "sources": data.get("sources") or {
                "primary": [],
                "secondary": [],
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _ensure_analysis_schema(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure analysis conforms to expected schema.
        """
        if not isinstance(analysis, dict):
            return self._create_fallback_analysis("Unknown")
        
        # Ensure all required keys exist
        defaults = {
            "market_positioning": {
                "segment": "B2B SaaS",
                "position": "Emerging",
                "market_share": "< 5%",
                "growth_trajectory": "Moderate",
                "competitive_advantages": []
            },
            "swot_analysis": {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            },
            "competitive_landscape": {
                "direct_competitors": [],
                "indirect_competitors": [],
                "competitive_moats": [],
                "vulnerability_areas": []
            },
            "strategic_recommendations": [],
            "threat_assessment": {
                "overall_threat_level": ThreatLevel.MEDIUM.value,
                "key_threat_factors": [],
                "mitigation_strategies": []
            },
            "confidence_score": 0.5,
            "analysis_date": datetime.now().isoformat()
        }
        
        for key, default_value in defaults.items():
            if key not in analysis or analysis[key] is None:
                analysis[key] = default_value
        
        return analysis
    
    def _create_fallback_company_data(
        self,
        competitor_name: str,
        error_msg: str = ""
    ) -> Dict[str, Any]:
        """
        Create minimal fallback company data.
        """
        return {
            "company_overview": {
                "name": competitor_name,
                "description": f"Unable to retrieve detailed information. {error_msg}",
                "industry": "Technology",
                "error": error_msg
            },
            "product_info": {
                "main_product": f"{competitor_name} Product",
                "key_features": ["Feature data unavailable"],
                "pricing_model": "Unknown",
                "target_market": ["Unknown"],
                "pricing": {}
            },
            "financial_data": {
                "revenue": "Data unavailable",
                "error": error_msg
            },
            "recent_news": [],
            "jobs": [],
            "reviews": {"average_rating": 0.0, "total_reviews": 0},
            "sources": {"error": [error_msg], "timestamp": datetime.now().isoformat()}
        }
    
    def _create_fallback_analysis(self, competitor_name: str) -> Dict[str, Any]:
        """
        Create safe fallback analysis when processing fails.
        """
        return {
            "market_positioning": {
                "segment": "Technology",
                "position": "Position under analysis",
                "market_share": "Data pending",
                "growth_trajectory": "To be determined",
                "competitive_advantages": ["Analysis in progress"]
            },
            "swot_analysis": {
                "strengths": ["Established presence"],
                "weaknesses": ["Limited data available"],
                "opportunities": ["Market expansion potential"],
                "threats": ["Competitive landscape"]
            },
            "competitive_landscape": {
                "direct_competitors": ["Analysis pending"],
                "indirect_competitors": ["Analysis pending"],
                "competitive_moats": ["To be determined"],
                "vulnerability_areas": ["Data collection in progress"]
            },
            "strategic_recommendations": [
                {
                    "category": "Data Collection",
                    "priority": "High",
                    "timeline": "Immediate",
                    "recommendations": ["Gather more comprehensive data for accurate analysis"],
                    "expected_impact": "Improved analysis accuracy"
                }
            ],
            "threat_assessment": {
                "overall_threat_level": ThreatLevel.MEDIUM.value,
                "key_threat_factors": ["Insufficient data for assessment"],
                "mitigation_strategies": ["Monitor competitor activities"]
            },
            "confidence_score": 0.5,
            "analysis_date": datetime.now().isoformat()
        }
    
    def _create_error_result(self, competitor_name: str, error_msg: str) -> AnalysisResult:
        """
        Create error result when critical failure occurs.
        """
        return AnalysisResult(
            competitor_name=competitor_name,
            company_info=self._create_fallback_company_data(competitor_name, error_msg),
            market_analysis=self._create_fallback_analysis(competitor_name),
            strategic_insights=[],
            confidence_score=0.0,
            coverage_metrics=CoverageMetrics(),
            generated_at=datetime.now().isoformat(),
            status=AnalysisStatus.ERROR,
            execution_time_sec=0.0,
            warnings=[f"⚠️ Critical error: {error_msg}"],
            metadata={"error": error_msg}
        )
    
    # ---------------------------------------------------------------------
    # Coverage Metrics
    # ---------------------------------------------------------------------
    
    def _calculate_coverage_metrics(self, company_data: Dict[str, Any]) -> CoverageMetrics:
        """
        Calculate comprehensive coverage metrics from company data.
        """
        metrics = CoverageMetrics()
        
        # Count news items
        news = company_data.get("recent_news", [])
        metrics.news_count = len(news) if isinstance(news, list) else 0
        
        # Count features
        features = (company_data.get("product_info", {}) or {}).get("key_features", [])
        metrics.features_count = len(features) if isinstance(features, list) else 0
        
        # Count sources
        sources = company_data.get("sources", {})
        if isinstance(sources, dict):
            all_sources = []
            for source_list in sources.values():
                if isinstance(source_list, list):
                    all_sources.extend(source_list)
            metrics.sources_count = len(set(all_sources))  # Unique sources
        
        # Check financial data presence
        financial = company_data.get("financial_data", {})
        if isinstance(financial, dict):
            metrics.financial_data_present = any(
                financial.get(key) and 
                str(financial.get(key)).lower() not in ["not disclosed", "unknown", "data unavailable", ""]
                for key in ["revenue", "funding", "valuation"]
            )
        
        # Check reviews presence
        reviews = company_data.get("reviews", {})
        if isinstance(reviews, dict):
            metrics.reviews_present = (
                reviews.get("total_reviews", 0) > 0 or
                reviews.get("average_rating", 0) > 0
            )
        
        # Count jobs
        jobs = company_data.get("jobs", [])
        metrics.jobs_count = len(jobs) if isinstance(jobs, list) else 0
        
        return metrics
    
    # ---------------------------------------------------------------------
    # Data Merging
    # ---------------------------------------------------------------------
    
    def _merge_company_data(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently merge new data into base while preserving schema stability.
        """
        merged = dict(base)
        
        # Merge overview (prefer new non-empty values)
        if "company_overview" in new:
            base_overview = merged.get("company_overview", {})
            new_overview = new.get("company_overview", {})
            merged["company_overview"] = self._merge_dicts(base_overview, new_overview)
        
        # Merge product info
        if "product_info" in new:
            base_product = merged.get("product_info", {})
            new_product = new.get("product_info", {})
            
            # Special handling for arrays
            merged_product = self._merge_dicts(base_product, new_product)
            
            # Merge features (avoiding duplicates)
            base_features = set(base_product.get("key_features", []))
            new_features = set(new_product.get("key_features", []))
            merged_product["key_features"] = list(base_features | new_features)
            
            merged["product_info"] = merged_product
        
        # Merge financial data
        if "financial_data" in new:
            base_financial = merged.get("financial_data", {})
            new_financial = new.get("financial_data", {})
            merged["financial_data"] = self._merge_dicts(base_financial, new_financial)
        
        # Merge news (keep unique by title)
        merged["recent_news"] = self._merge_news_items(
            base.get("recent_news", []),
            new.get("recent_news", [])
        )
        
        # Merge jobs (keep unique by title)
        merged["jobs"] = self._merge_job_items(
            base.get("jobs", []),
            new.get("jobs", [])
        )
        
        # Override reviews if new has data
        new_reviews = new.get("reviews", {})
        if new_reviews and new_reviews.get("total_reviews", 0) > 0:
            merged["reviews"] = new_reviews
        
        # Merge sources
        merged["sources"] = self._merge_sources(
            base.get("sources", {}),
            new.get("sources", {})
        )
        
        return merged
    
    def _merge_dicts(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two dictionaries, preferring non-empty new values.
        """
        result = dict(base)
        for key, value in new.items():
            if value and (
                value not in ["", "Unknown", "Not disclosed", "Data unavailable"] or
                key not in result or
                not result[key]
            ):
                result[key] = value
        return result
    
    def _merge_news_items(self, base: List, new: List) -> List:
        """
        Merge news items, avoiding duplicates by title.
        """
        seen_titles = set()
        merged = []
        
        for item in base + new:
            if isinstance(item, dict):
                title = item.get("title", "").strip().lower()
                if title and title not in seen_titles:
                    merged.append(item)
                    seen_titles.add(title)
            elif isinstance(item, str):
                if item not in seen_titles:
                    merged.append(item)
                    seen_titles.add(item)
        
        return merged[:20]  # Limit to 20 most recent
    
    def _merge_job_items(self, base: List, new: List) -> List:
        """
        Merge job items, avoiding duplicates.
        """
        seen = set()
        merged = []
        
        for item in base + new:
            if isinstance(item, dict):
                key = (item.get("title", ""), item.get("location", ""))
                if key not in seen:
                    merged.append(item)
                    seen.add(key)
            else:
                if str(item) not in seen:
                    merged.append(item)
                    seen.add(str(item))
        
        return merged[:15]  # Limit to 15 jobs
    
    def _merge_sources(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge source dictionaries.
        """
        merged = {}
        all_keys = set(base.keys()) | set(new.keys())
        
        for key in all_keys:
            base_list = base.get(key, [])
            new_list = new.get(key, [])
            
            if isinstance(base_list, list) and isinstance(new_list, list):
                # Merge and deduplicate
                merged[key] = list(set(base_list) | set(new_list))
            else:
                # Prefer new if exists, otherwise base
                merged[key] = new_list if new_list else base_list
        
        # Add timestamp
        merged["timestamp"] = datetime.now().isoformat()
        
        return merged
    
    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    
    def _create_default_callbacks(self) -> UICallbacks:
        """
        Create no-op callbacks when none provided.
        """
        class NoOpCallbacks:
            def on_progress(self, percent: int, message: str) -> None:
                logger.debug(f"Progress: {percent}% - {message}")
            
            def on_done(self) -> None:
                logger.debug("Analysis complete")
            
            def on_error(self, error: str) -> None:
                logger.error(f"Error callback: {error}")
        
        return NoOpCallbacks()
    
    def _safe_progress(self, percent: int, message: str) -> None:
        """
        Safely call progress callback.
        """
        try:
            percent = max(0, min(100, int(percent)))
            self._ui.on_progress(percent, str(message))
        except Exception as e:
            logger.debug(f"Progress callback error (non-critical): {e}")
    
    def _safe_done(self) -> None:
        """
        Safely call done callback.
        """
        try:
            self._ui.on_done()
        except Exception as e:
            logger.debug(f"Done callback error (non-critical): {e}")
    
    def _safe_error(self, error: str) -> None:
        """
        Safely call error callback if available.
        """
        try:
            if hasattr(self._ui, 'on_error'):
                self._ui.on_error(error)
        except Exception as e:
            logger.debug(f"Error callback error (non-critical): {e}")
    
    def _log_context(self, **kwargs):
        """
        Context manager for structured logging (placeholder for actual implementation).
        """
        class LogContext:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return LogContext()


# ---------------------------
# Helper Functions
# ---------------------------

def create_orchestrator_from_config(
    data_agent: DataAgent,
    analysis_engine: AnalysisAgent,
    config_dict: Optional[Dict[str, Any]] = None,
    ui_callbacks: Optional[UICallbacks] = None
) -> CompeteAIOrchestrator:
    """
    Factory function to create orchestrator from configuration dictionary.
    
    Parameters
    ----------
    data_agent : DataAgent
        Data collection agent
    analysis_engine : AnalysisAgent
        Analysis engine
    config_dict : dict, optional
        Configuration overrides
    ui_callbacks : UICallbacks, optional
        UI callbacks
        
    Returns
    -------
    CompeteAIOrchestrator
        Configured orchestrator instance
    """
    config = OrchestratorConfig()
    
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return CompeteAIOrchestrator(
        data_agent=data_agent,
        analysis_engine=analysis_engine,
        ui_callbacks=ui_callbacks,
        config=config
    )


# ---------------------------
# Export
# ---------------------------

__all__ = [
    'CompeteAIOrchestrator',
    'OrchestratorConfig',
    'AnalysisStatus',
    'AnalysisResult',
    'CoverageMetrics',
    'UICallbacks',
    'DataAgent',
    'AnalysisAgent',
    'create_orchestrator_from_config',
    'ThreatLevel'
]