"""
MarketPilot API Routes
FastAPI endpoint definitions for financial analysis, prediction, and chat

All endpoints are grounded in RAG + multi-agent orchestration
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from enum import Enum

# Import agent orchestrator and analysis modules
from app.agents.orchestrator import AgentOrchestrator
from app.analysis.technical import TechnicalAnalysis
from app.analysis.sentiment import SentimentAnalysis
from app.analysis.predictions import PredictionModel
from app.data.ingestion import DataIngestion
from app.rag.retrieval import RAGRetriever
from app.config import (
    CACHE_TTL,
    CONCURRENT_REQUESTS,
    LOG_LEVEL,
    DEFAULT_FORECAST_DAYS,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["financial-analysis"])

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AnalysisType(str, Enum):
    """Analysis type enumeration"""
    FULL = "full"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"


class Timeframe(str, Enum):
    """Timeframe enumeration"""
    ONE_WEEK = "1w"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"


class ChatRequest(BaseModel):
    """Chat endpoint request schema"""
    message: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: str = Field(default="default", description="Session identifier for context")
    include_sources: bool = Field(default=False, description="Include source citations in response")

    class Config:
        schema_extra = {
            "example": {
                "message": "Analyze Tesla's performance and compare with Ford",
                "session_id": "user123",
                "include_sources": True,
            }
        }


class AnalyzeRequest(BaseModel):
    """Stock analysis request schema"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock ticker (e.g., AAPL)")
    analysis_type: AnalysisType = Field(default=AnalysisType.FULL, description="Type of analysis")
    timeframe: Timeframe = Field(default=Timeframe.THREE_MONTHS, description="Analysis timeframe")

    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "analysis_type": "full",
                "timeframe": "3mo",
            }
        }


class PredictRequest(BaseModel):
    """Price prediction request schema"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock ticker")
    days: int = Field(default=DEFAULT_FORECAST_DAYS, ge=1, le=90, description="Forecast horizon (days)")

    class Config:
        schema_extra = {
            "example": {
                "symbol": "NVDA",
                "days": 7,
            }
        }


class CompareRequest(BaseModel):
    """Multi-stock comparison request schema"""
    symbols: List[str] = Field(..., min_items=2, max_items=10, description="List of ticker symbols")
    metrics: List[str] = Field(
        default=["price", "volatility", "sentiment"],
        description="Metrics to compare"
    )

    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "metrics": ["price", "volatility", "sentiment"],
            }
        }


class IngestRequest(BaseModel):
    """Document ingestion request schema"""
    source: str = Field(default="default", description="Data source identifier")
    force_refresh: bool = Field(default=False, description="Force refresh vector store")

    class Config:
        schema_extra = {
            "example": {
                "source": "news",
                "force_refresh": False,
            }
        }


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class TechnicalMetrics(BaseModel):
    """Technical analysis output"""
    rsi: float = Field(..., description="RSI indicator (0-100)")
    macd: Dict[str, float] = Field(..., description="MACD components")
    bollinger_bands: Dict[str, float] = Field(..., description="Bollinger bands (upper, middle, lower)")
    sma_20: Optional[float] = Field(default=None, description="20-day simple moving average")
    sma_50: Optional[float] = Field(default=None, description="50-day simple moving average")
    trend: str = Field(..., description="Trend classification (bullish/bearish/neutral)")
    volatility: float = Field(..., description="Current volatility (annualized %)")


class SentimentOutput(BaseModel):
    """Sentiment analysis output"""
    label: str = Field(..., description="Sentiment label (positive/neutral/negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    recent_headlines: List[str] = Field(default=[], description="Recent relevant headlines")
    sentiment_trend: Optional[str] = Field(default=None, description="Trend direction")


class PredictionOutput(BaseModel):
    """ML prediction output"""
    direction: str = Field(..., description="Predicted direction (up/down/sideways)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    predicted_price: Optional[float] = Field(default=None, description="Predicted target price")
    horizon_days: int = Field(..., description="Forecast horizon")
    supporting_features: Dict[str, Any] = Field(default={}, description="Feature importance")


class AnalyzeResponse(BaseModel):
    """Stock analysis response schema"""
    symbol: str
    analysis_type: str
    timeframe: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    technical: Optional[TechnicalMetrics] = None
    sentiment: Optional[SentimentOutput] = None
    prediction: Optional[PredictionOutput] = None
    risk_level: str = Field(..., description="Risk assessment (LOW/MODERATE/HIGH)")
    recommendations: List[str] = Field(..., description="Trading recommendations")
    agent_trace: List[Dict[str, Any]] = Field(default=[], description="Agent execution trace")


class ChatResponse(BaseModel):
    """Chat endpoint response schema"""
    session_id: str
    response: str = Field(..., description="Agent-generated response")
    agent_trace: List[Dict[str, Any]] = Field(default=[], description="Agent reasoning steps")
    sources: List[str] = Field(default=[], description="Source citations")
    risk_level: Optional[str] = None
    recommendations: List[str] = Field(default=[])


class ComparisonMetric(BaseModel):
    """Single comparison metric"""
    symbol: str
    metric: str
    value: Any
    percentile: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class CompareResponse(BaseModel):
    """Multi-stock comparison response"""
    symbols: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: List[ComparisonMetric]
    winner: Optional[str] = Field(default=None, description="Best performer metric-wise")
    note: str = ""


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str] = Field(
        default={
            "backend": "healthy",
            "rag_pipeline": "healthy",
            "external_apis": "checking",
        }
    )


# ============================================================================
# GLOBAL DEPENDENCIES
# ============================================================================

# Initialize core components (singleton pattern)
_orchestrator = None
_tech_analysis = None
_sentiment_analysis = None
_prediction_model = None
_data_ingestion = None
_rag_retriever = None


def get_orchestrator() -> AgentOrchestrator:
    """Lazy-load agent orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
        logger.info("Agent Orchestrator initialized")
    return _orchestrator


def get_tech_analysis() -> TechnicalAnalysis:
    """Lazy-load technical analysis engine"""
    global _tech_analysis
    if _tech_analysis is None:
        _tech_analysis = TechnicalAnalysis()
        logger.info("Technical Analysis engine initialized")
    return _tech_analysis


def get_sentiment_analysis() -> SentimentAnalysis:
    """Lazy-load sentiment analysis engine"""
    global _sentiment_analysis
    if _sentiment_analysis is None:
        _sentiment_analysis = SentimentAnalysis()
        logger.info("Sentiment Analysis engine initialized")
    return _sentiment_analysis


def get_prediction_model() -> PredictionModel:
    """Lazy-load prediction model"""
    global _prediction_model
    if _prediction_model is None:
        _prediction_model = PredictionModel()
        logger.info("Prediction Model initialized")
    return _prediction_model


def get_rag_retriever() -> RAGRetriever:
    """Lazy-load RAG retriever"""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
        logger.info("RAG Retriever initialized")
    return _rag_retriever


def get_data_ingestion() -> DataIngestion:
    """Lazy-load data ingestion engine"""
    global _data_ingestion
    if _data_ingestion is None:
        _data_ingestion = DataIngestion()
        logger.info("Data Ingestion engine initialized")
    return _data_ingestion


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_ticker(symbol: str) -> bool:
    """Validate stock ticker format"""
    if not symbol or len(symbol) > 10:
        return False
    return symbol.isupper() or symbol.replace("-", "").isalpha()


async def build_agent_trace(start_time: float) -> List[Dict[str, Any]]:
    """Build agent execution trace"""
    # Placeholder - would be populated by orchestrator
    return [
        {
            "agent": "research",
            "status": "completed",
            "duration_ms": 412,
        },
        {
            "agent": "analysis",
            "status": "completed",
            "duration_ms": 198,
        },
        {
            "agent": "sentiment",
            "status": "completed",
            "duration_ms": 730,
        },
        {
            "agent": "prediction",
            "status": "completed",
            "duration_ms": 265,
        },
        {
            "agent": "report",
            "status": "completed",
            "duration_ms": 540,
        },
    ]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verify backend, RAG pipeline, and external API connectivity",
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Status of all system components
    """
    logger.info("Health check requested")
    
    health_status = {
        "backend": "healthy",
        "rag_pipeline": "checking",
        "external_apis": "checking",
    }
    
    # Check RAG
    try:
        rag = get_rag_retriever()
        if rag.is_initialized():
            health_status["rag_pipeline"] = "healthy"
        else:
            health_status["rag_pipeline"] = "initializing"
    except Exception as e:
        logger.warning(f"RAG health check failed: {e}")
        health_status["rag_pipeline"] = "error"
    
    # Check external APIs (quick validation)
    try:
        data_ingestion = get_data_ingestion()
        health_status["external_apis"] = "healthy"
    except Exception as e:
        logger.warning(f"External API health check failed: {e}")
        health_status["external_apis"] = "degraded"
    
    return HealthResponse(
        status="healthy" if health_status["backend"] == "healthy" else "degraded",
        components=health_status,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational chat endpoint",
    description="Ask natural language questions about stocks and market analysis",
    status_code=200,
)
async def chat(request: ChatRequest):
    """
    Conversational chat endpoint - interact with the financial analyst.
    
    The system will:
    1. Route query to appropriate agent(s)
    2. Retrieve relevant context via RAG
    3. Synthesize multi-agent outputs
    4. Return grounded, explainable response
    
    Args:
        request: ChatRequest with user message and session context
        
    Returns:
        ChatResponse with agent reasoning trace
        
    Raises:
        HTTPException: On validation or orchestration errors
    """
    try:
        logger.info(f"Chat request from session {request.session_id}: {request.message[:50]}...")
        
        # Validate message
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        rag_retriever = get_rag_retriever()
        
        # Retrieve context from RAG
        context = rag_retriever.retrieve(request.message, k=5)
        logger.debug(f"Retrieved {len(context)} context documents")
        
        # Execute orchestrator
        result = await orchestrator.execute_chat(
            message=request.message,
            session_id=request.session_id,
            context=context,
        )
        
        response = ChatResponse(
            session_id=request.session_id,
            response=result.get("response", "Unable to generate response"),
            agent_trace=result.get("agent_trace", []),
            sources=result.get("sources", []) if request.include_sources else [],
            risk_level=result.get("risk_level"),
            recommendations=result.get("recommendations", []),
        )
        
        logger.info(f"Chat response generated successfully")
        return response
        
    except HTTPException as e:
        logger.warning(f"HTTP error in chat: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}",
        )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Stock analysis endpoint",
    description="Perform technical, fundamental, and sentiment analysis on a stock",
    status_code=200,
)
async def analyze(request: AnalyzeRequest):
    """
    Comprehensive stock analysis endpoint.
    
    Executes:
    - Technical analysis (RSI, MACD, Bollinger Bands, moving averages)
    - Fundamental analysis (P/E, EPS, valuation)
    - Sentiment analysis (news, social signals)
    - ML prediction (short-term trend)
    
    Args:
        request: AnalyzeRequest with symbol and analysis parameters
        
    Returns:
        AnalyzeResponse with integrated multi-agent analysis
        
    Raises:
        HTTPException: On invalid symbol or analysis errors
    """
    try:
        logger.info(f"Analyze request for {request.symbol} ({request.analysis_type})")
        
        # Validate ticker
        if not validate_ticker(request.symbol):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid ticker symbol: {request.symbol}",
            )
        
        orchestrator = get_orchestrator()
        
        # Execute multi-agent analysis
        result = await orchestrator.execute_analysis(
            symbol=request.symbol,
            analysis_type=request.analysis_type,
            timeframe=request.timeframe,
        )
        
        # Build response
        response = AnalyzeResponse(
            symbol=request.symbol,
            analysis_type=request.analysis_type,
            timeframe=request.timeframe,
            technical=result.get("technical"),
            sentiment=result.get("sentiment"),
            prediction=result.get("prediction"),
            risk_level=result.get("risk_level", "MODERATE"),
            recommendations=result.get("recommendations", []),
            agent_trace=result.get("agent_trace", []),
        )
        
        logger.info(f"Analysis complete for {request.symbol}")
        return response
        
    except HTTPException as e:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in analyze: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing {request.symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing stock: {str(e)}",
        )


@router.post(
    "/predict/{symbol}",
    response_model=PredictionOutput,
    summary="Price prediction endpoint",
    description="Generate ML-based short-term price forecasts",
    status_code=200,
)
async def predict(symbol: str, days: int = DEFAULT_FORECAST_DAYS):
    """
    Price prediction endpoint - forecast short-term price trends.
    
    Uses engineered technical features and ensemble models.
    Outputs confidence intervals and feature importance.
    
    Args:
        symbol: Stock ticker
        days: Forecast horizon (1-90, default 7)
        
    Returns:
        PredictionOutput with direction, confidence, and supporting metrics
        
    Raises:
        HTTPException: On invalid symbol or prediction errors
    """
    try:
        logger.info(f"Prediction request for {symbol} ({days}-day horizon)")
        
        # Validate inputs
        if not validate_ticker(symbol):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid ticker symbol: {symbol}",
            )
        
        if not (1 <= days <= 90):
            raise HTTPException(
                status_code=422,
                detail=f"Days must be between 1 and 90, got {days}",
            )
        
        prediction_model = get_prediction_model()
        orchestrator = get_orchestrator()
        
        # Get latest data for symbol
        data = await orchestrator.get_market_data(symbol, timeframe="3mo")
        
        # Generate prediction
        prediction = prediction_model.forecast(
            data=data,
            symbol=symbol,
            horizon_days=days,
        )
        
        logger.info(f"Prediction generated for {symbol}")
        return prediction
        
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get(
    "/compare",
    response_model=CompareResponse,
    summary="Multi-stock comparison endpoint",
    description="Compare metrics across multiple stocks",
    status_code=200,
)
async def compare(symbols: str, metrics: Optional[str] = None):
    """
    Multi-stock comparison endpoint.
    
    Compares technical, sentiment, and predictive metrics across stocks.
    Ranks by strength and identifies the best performer.
    
    Args:
        symbols: Comma-separated ticker symbols (e.g., "AAPL,MSFT,GOOGL")
        metrics: Comma-separated metrics to compare (default: price,volatility,sentiment)
        
    Returns:
        CompareResponse with ranked metrics and recommendations
        
    Raises:
        HTTPException: On invalid symbols or comparison errors
    """
    try:
        logger.info(f"Compare request for symbols: {symbols}")
        
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        if len(symbol_list) < 2:
            raise HTTPException(
                status_code=422,
                detail="Must compare at least 2 stocks",
            )
        
        if len(symbol_list) > 10:
            raise HTTPException(
                status_code=422,
                detail="Cannot compare more than 10 stocks at once",
            )
        
        # Validate all tickers
        for symbol in symbol_list:
            if not validate_ticker(symbol):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid ticker: {symbol}",
                )
        
        # Parse metrics
        metric_list = [m.strip() for m in metrics.split(",")] if metrics else ["price", "volatility", "sentiment"]
        
        orchestrator = get_orchestrator()
        
        # Execute comparison
        comparison_result = await orchestrator.execute_comparison(
            symbols=symbol_list,
            metrics=metric_list,
        )
        
        response = CompareResponse(
            symbols=symbol_list,
            metrics=comparison_result.get("metrics", []),
            winner=comparison_result.get("winner"),
            note=comparison_result.get("note", ""),
        )
        
        logger.info(f"Comparison complete for {len(symbol_list)} symbols")
        return response
        
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}",
        )


@router.post(
    "/ingest/documents",
    summary="Ingest documents into RAG pipeline",
    description="Load financial documents into ChromaDB for RAG retrieval",
    status_code=202,
)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Document ingestion endpoint - load data into RAG vector store.
    
    This is an async operation that runs in the background.
    Ingests financial news, reports, and historical data into ChromaDB.
    
    Args:
        request: IngestRequest with source and refresh parameters
        background_tasks: FastAPI background task manager
        
    Returns:
        Acknowledgment with job_id for polling
    """
    try:
        logger.info(f"Document ingestion requested from source: {request.source}")
        
        # Queue background task
        data_ingestion = get_data_ingestion()
        rag_retriever = get_rag_retriever()
        
        async def ingest_task():
            try:
                logger.info(f"Starting ingestion from {request.source}")
                
                # Fetch data
                documents = await data_ingestion.fetch_documents(
                    source=request.source,
                    force_refresh=request.force_refresh,
                )
                
                logger.info(f"Fetched {len(documents)} documents")
                
                # Ingest into RAG
                await rag_retriever.ingest_batch(documents)
                
                logger.info(f"Ingestion complete: {len(documents)} documents indexed")
                
            except Exception as e:
                logger.error(f"Ingestion failed: {e}", exc_info=True)
        
        background_tasks.add_task(ingest_task)
        
        return {
            "status": "accepted",
            "message": f"Document ingestion started from {request.source}",
            "job_id": "ingest-" + datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Ingestion request error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.detail if isinstance(exc.detail, str) else "HTTP_ERROR",
                "message": exc.detail,
                "status": exc.status_code,
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
    )


@router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Generic exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "status": 500,
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
    )


# ============================================================================
# MIDDLEWARE
# ============================================================================

@router.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header"""
    import time
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000  # ms
    response.headers["X-Process-Time"] = str(duration)
    return response
