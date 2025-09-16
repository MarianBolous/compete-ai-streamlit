# CompeteAI - Advanced Competitive Intelligence Platform

AI-powered competitive analysis tool built with Streamlit, featuring multi-agent AI architecture for comprehensive competitor research and strategic insights.

## Features

- **Multi-Agent AI System**: Combines OpenAI GPT-4 and Anthropic Claude
- **Real-time Analysis**: Automated competitive intelligence gathering
- **Strategic Insights**: Actionable recommendations with timelines
- **Interactive Visualizations**: Market positioning and competitive analysis charts
- **Professional UI**: Modern, responsive interface with advanced styling

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the app**: Open http://localhost:8501 in your browser

## Configuration

- Enter API keys in the sidebar for live AI analysis
- Use Demo Mode for testing with mock data
- Configure company context for personalized insights

## Architecture

The application uses a sophisticated multi-agent AI system:
- **WebIntelligenceAgent**: Gathers competitive data
- **MarketAnalysisAgent**: Analyzes market positioning
- **StrategicInsightsAgent**: Generates recommendations
- **AI Orchestrator**: Coordinates all agents

## Deployment

Compatible with:
- Local development
- Streamlit Cloud
- Heroku
- Docker containers

## API Keys Required

- OpenAI API Key: https://platform.openai.com/api-keys
- Anthropic API Key: https://console.anthropic.com/
