# CompeteAI - Competitive Intelligence Platform

A production-ready competitive intelligence platform that provides real-time analysis of competitors using AI-powered multi-agent systems with comprehensive guardrails and multi-model API integration.

## 🚀 Features

### Core Intelligence Capabilities
- **Multi-Agent Architecture**: Specialized agents for data collection and AI analysis with orchestrated workflows
- **Real-time Competitive Analysis**: Live intelligence gathering with iterative refinement and coverage thresholds
- **Confidence Scoring System**: Advanced reliability metrics (0-100%) with section-level confidence tracking
- **Delta Tracking & Change Detection**: Compare analysis runs with detailed change summaries and impact assessment
- **SWOT Analysis Engine**: Automated strengths, weaknesses, opportunities, and threats identification
- **Threat Assessment Matrix**: Multi-dimensional threat evaluation with mitigation strategies
- **Strategic Recommendations**: AI-generated actionable insights across product, market, competitive, and financial strategies

### Data Collection & Sources
- **Multi-Source Data Aggregation**: News, pricing, features, job postings, customer reviews, and financial signals
- **Source Attribution**: Full traceability with confidence scores per data source
- **Iterative Data Refinement**: "Stop-when-satisfied" collection with coverage thresholds
- **Schema Stability**: Consistent data structures with graceful degradation
- **Real-time Progress Tracking**: Live updates during data collection and analysis phases

### User Experience
- **Interactive Dashboard**: Modern Streamlit-based UI with real-time progress indicators
- **Comprehensive Visualizations**: Charts, metrics, and comparative analysis displays
- **Export Capabilities**: JSON, structured data, and report generation
- **Responsive Design**: Mobile-friendly interface with intuitive navigation
- **Error Handling**: User-friendly error messages with recovery suggestions

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

## 🏗️ Competitive Architecture

### Multi-Agent Orchestration Pattern
The platform implements a sophisticated **Plan → Delegate → Evaluate → Iterate → Finalize** architecture:

- **Orchestrator (Agentic Conductor)**: Central coordinator managing agent workflows with timeout handling and loop evaluation
- **Data Collection Agent**: Specialized intelligence gathering with multi-tool architecture and iterative refinement
- **AI Analysis Engine**: Advanced processing using multi-model routing and confidence scoring
- **UI Layer**: Real-time interface with progress callbacks and error handling

### Key Architectural Decisions

#### 1. **Agent Delegation Strategy**
- **Separation of Concerns**: Data collection and analysis are handled by specialized agents
- **Asynchronous Execution**: Non-blocking operations with timeout management
- **Graceful Degradation**: Partial results when components fail
- **Schema Stability**: Consistent data structures across all operations

#### 2. **Iterative Refinement Engine**
- **Coverage Thresholds**: "Stop-when-satisfied" criteria with configurable minimums
- **Confidence-Based Evaluation**: Dynamic quality assessment with retry logic
- **Loop Protection**: Maximum iteration limits with evaluation policies
- **Delta Tracking**: Change detection between analysis runs

#### 3. **Production-Ready Design**
- **Comprehensive Guardrails**: Timeouts, retries, rate limiting, and caching
- **Multi-Model Fallbacks**: OpenAI/Anthropic routing with automatic failover
- **Error Recovery**: Partial results and user-friendly error handling
- **Performance Optimization**: Caching, connection pooling, and resource management

## 🛡️ Comprehensive Guardrails

### Timeout & Resource Management
- **Per-Agent Timeouts**: Configurable timeouts for data collection (15s) and analysis (45s)
- **Non-Blocking Fallbacks**: Graceful degradation when components exceed time limits
- **Resource Pooling**: HTTP connection pooling with configurable pool sizes
- **Memory Management**: Automatic cleanup and resource deallocation

### Rate Limiting & API Protection
- **Token Bucket Rate Limiter**: Per-host rate limiting (30 calls/60 seconds)
- **Exponential Backoff**: Intelligent retry strategies with configurable backoff factors
- **API Key Rotation**: Support for multiple API keys with automatic failover
- **Request Throttling**: Adaptive throttling based on API response patterns

### Error Handling & Recovery
- **Comprehensive Retry Logic**: Handles connection, read, and status errors (429, 5xx)
- **Circuit Breaker Pattern**: Prevents cascade failures with automatic recovery
- **Partial Result Handling**: Returns useful data even when some components fail
- **User-Friendly Error Messages**: Clear error communication with recovery suggestions

### Caching & Performance
- **SQLite-Based Caching**: Persistent caching with configurable TTL (24 hours)
- **Embedding Cache**: Optimized similarity calculations with cache hits
- **Request Deduplication**: Prevents duplicate API calls within time windows
- **Progressive Loading**: Incremental data loading with real-time progress updates

### Data Quality & Validation
- **Schema Validation**: Ensures data consistency across all operations
- **Confidence Scoring**: Per-section and overall confidence metrics
- **Source Attribution**: Full traceability of data sources and collection methods
- **Coverage Analysis**: Ensures minimum data requirements are met

## 🤖 Multi-Model API Integration

### Intelligent Model Routing
- **Per-Stage Preferences**: Configure different models for different analysis stages
- **Automatic Fallback**: OpenAI → Anthropic fallback chain with error handling
- **Model-Specific Optimization**: Tailored prompts and parameters per model
- **Cost Optimization**: Smart routing based on task complexity and model costs

### Supported Models
- **OpenAI**: GPT-4o-mini (default), GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-Sonnet (default), Claude-3-Opus, Claude-3-Haiku
- **Configuration**: Per-stage model selection with temperature and token controls

### Advanced Features
- **Temperature Control**: Configurable creativity levels per analysis stage
- **Token Management**: Intelligent token allocation and response truncation
- **Prompt Engineering**: Optimized prompts for each model's strengths
- **Response Validation**: JSON schema validation and error recovery

## 📊 Data Collection & Scraping

### Multi-Source Intelligence Gathering
- **News Intelligence**: Real-time news aggregation with sentiment analysis
- **Pricing Intelligence**: Competitive pricing models and feature comparisons
- **Feature Analysis**: Product capability mapping and gap identification
- **Job Market Signals**: Hiring trends and organizational growth indicators
- **Customer Reviews**: Sentiment analysis and satisfaction metrics
- **Financial Signals**: Funding, revenue, and growth trajectory analysis

### Data Collection Tools
- **News Tool**: Multi-provider news aggregation (NewsAPI, Google News, SerpAPI)
- **Pricing Tool**: Automated pricing page analysis and feature extraction
- **Features Tool**: Product documentation and capability mapping
- **Jobs Tool**: Employment data and organizational insights
- **Reviews Tool**: Customer feedback aggregation and sentiment scoring
- **Financial Tool**: Investment and revenue intelligence

### Quality Assurance
- **Source Verification**: Multiple source cross-validation
- **Freshness Tracking**: Data age and update frequency monitoring
- **Completeness Scoring**: Coverage analysis across all data dimensions
- **Bias Detection**: Multi-source validation to reduce single-source bias

## Deployment

Compatible with:
- Local development
- Streamlit Cloud
- Heroku
- Docker containers

## API Keys Required

- OpenAI API Key: https://platform.openai.com/api-keys
- Anthropic API Key: https://console.anthropic.com/
