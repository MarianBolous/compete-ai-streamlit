import logging
logging.basicConfig(level=logging.INFO)

from agents.analysis_engine import create_analysis_engine, AnalysisConfig

engine = create_analysis_engine(
    config=AnalysisConfig(
        enable_llm_refinement=False  # keep off for first run
    )
)

company_name = "TestCo"
company_data = {
    "product_info": {"key_features": ["AI insights", "API access", "Dashboards"]},
    "financial_data": {"total_funding": "50M", "growth_rate": "60% YoY"},
    "recent_news": ["Launch X", "Partnership Y", "Award Z"],
    "reviews": {"avg_rating": 4.2}
}
user_context = {"industry": "SaaS", "company_stage": "Growth Stage"}

result = engine.analyze_competitor(company_name, company_data, user_context)
is_valid, issues = (True, [])
try:
    from agents.analysis_engine import validate_analysis_result
    is_valid, issues = validate_analysis_result(result)
except Exception as e:
    print("Validation import failed (non-fatal):", e)

print("Status:", result.status)
print("Confidence:", result.confidence_score)
print("Position:", result.market_positioning.get("position"))
print("SWOT strengths:", result.swot_analysis.get("strengths"))
print("Valid?", is_valid, "| Issues:", issues)
