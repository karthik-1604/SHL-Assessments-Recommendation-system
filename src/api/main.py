#!/usr/bin/env python3
"""
FastAPI Application for SHL Assessment Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.recommender import SHLRecommender

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Intelligent recommendation system for SHL assessments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender = None

# Pydantic models for API
class RecommendationRequest(BaseModel):
    query: str

class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]

class HealthResponse(BaseModel):
    status: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the recommender system on startup"""
    global recommender
    
    logger.info("🚀 Starting SHL Assessment Recommendation API")
    
    try:
        recommender = SHLRecommender()
        
        logger.info("⏳ Initializing recommender system...")
        start_time = time.time()
        
        if recommender.initialize():
            init_time = time.time() - start_time
            logger.info(f"✅ Recommender system initialized in {init_time:.2f} seconds")
        else:
            logger.error("❌ Failed to initialize recommender system")
            recommender = None
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        recommender = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender system not initialized")
    
    return HealthResponse(status="healthy")

# Main recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: RecommendationRequest):
    """
    Get assessment recommendations based on query
    
    Args:
        request: Contains the query string
        
    Returns:
        List of recommended assessments with metadata
    """
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender system not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"📥 Received recommendation request: '{request.query}'")
        start_time = time.time()
        
        # Get recommendations (now returns dict with 'recommended_assessments' key)
        result = recommender.recommend(request.query, top_k=10)
        recommendations = result.get('recommended_assessments', [])
        
        processing_time = time.time() - start_time
        logger.info(f"⚡ Processed in {processing_time:.2f} seconds")
        
        # Ensure we have at least 1 and at most 10 recommendations
        if len(recommendations) == 0:
            logger.warning("⚠️ No recommendations found, returning fallback")
            # Return a fallback recommendation if none found
            fallback = {
                'url': 'https://www.shl.com/solutions/products/product-catalog/',
                'name': 'SHL Product Catalog',
                'adaptive_support': 'No',
                'description': 'Browse the full SHL assessment catalog for suitable options.',
                'duration': 0,
                'remote_support': 'Yes',
                'test_type': ['General']
            }
            recommendations = [fallback]
        
        recommendations = recommendations[:10]  # Ensure max 10
        
        # Convert to response format (already in correct format)
        assessment_responses = []
        for rec in recommendations:
            assessment_response = AssessmentResponse(
                url=rec.get('url', ''),
                name=rec.get('name', ''),
                adaptive_support=rec.get('adaptive_support', 'No'),
                description=rec.get('description', ''),
                duration=rec.get('duration', 0),
                remote_support=rec.get('remote_support', 'No'),
                test_type=rec.get('test_type', [])
            )
            assessment_responses.append(assessment_response)
        
        logger.info(f"✅ Returning {len(assessment_responses)} recommendations")
        
        return RecommendationResponse(recommended_assessments=assessment_responses)
        
    except Exception as e:
        logger.error(f"❌ Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend"
        },
        "status": "online" if recommender is not None else "initializing"
    }

# Additional utility endpoint for testing
@app.get("/status")
async def get_status():
    """Get detailed system status"""
    
    if recommender is None:
        return {
            "status": "not_initialized",
            "message": "Recommender system not ready"
        }
    
    return {
        "status": "ready",
        "message": "Recommender system is operational",
        "assessments_loaded": len(recommender.assessments) if hasattr(recommender, 'assessments') else 0,
        "model": recommender.model_name if hasattr(recommender, 'model_name') else "unknown"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    logger.info("🚀 Starting SHL Assessment Recommendation API Server")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )