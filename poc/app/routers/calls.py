import os
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional, List

from ..models.database import get_db, CallAnalysis
from ..models.schemas import CallAnalysisResult, CallType
from ..services.transcription import transcription_service
from ..services.sentiment_analysis import sentiment_service

router = APIRouter(prefix="/api/calls", tags=["calls"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")


@router.post("/upload", response_model=CallAnalysisResult)
async def upload_and_analyze_call(
    file: UploadFile = File(...),
    agent_id: Optional[str] = Form(None),
    agent_name: Optional[str] = Form(None),
    customer_name: Optional[str] = Form(None),
    customer_phone: Optional[str] = Form(None),
    call_type: CallType = Form(CallType.INCOMING),
    db: Session = Depends(get_db)
):
    """
    Upload a call recording, transcribe it using Whisper, and analyze sentiment using GPT-3.5
    """
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/m4a", "audio/webm", "audio/ogg"]
    if file.content_type not in allowed_types and not file.filename.endswith(('.mp3', '.wav', '.m4a', '.webm', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: mp3, wav, m4a, webm, ogg")
    
    # Generate unique call ID
    call_id = str(uuid.uuid4())
    
    # Save uploaded file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_extension = os.path.splitext(file.filename)[1] or ".mp3"
    file_path = os.path.join(UPLOAD_DIR, f"{call_id}{file_extension}")
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        # Step 1: Transcribe using Whisper
        transcription_result = await transcription_service.transcribe_audio(file_path)
        
        # Step 2: Analyze using GPT-3.5
        analysis_result = await sentiment_service.analyze_call(transcription_result["text"])
        
        # Calculate total scores
        total_score = sum(q.score for q in analysis_result["question_scores"])
        max_score = sum(q.max_score for q in analysis_result["question_scores"])
        overall_percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Create result object
        result = CallAnalysisResult(
            call_id=call_id,
            call_date=datetime.utcnow(),
            audit_date=datetime.utcnow(),
            duration_seconds=transcription_result.get("duration", 0),
            agent_id=agent_id,
            agent_name=agent_name,
            customer_name=customer_name,
            customer_phone=customer_phone,
            transcription=transcription_result["text"],
            language=transcription_result.get("language", "unknown"),
            call_summary=analysis_result["call_summary"],
            customer_sentiment=analysis_result["customer_sentiment"],
            agent_behavior=analysis_result["agent_behavior"],
            compliance_risk=analysis_result["compliance_risk"],
            question_scores=analysis_result["question_scores"],
            total_score=total_score,
            max_score=max_score,
            overall_percentage=round(overall_percentage, 2),
            customer_intent=analysis_result["customer_intent"],
            key_issues=analysis_result["key_issues"],
            resolution_status=analysis_result["resolution_status"],
            follow_up_required=analysis_result["follow_up_required"]
        )
        
        # Save to database
        db_record = CallAnalysis(
            id=call_id,
            call_date=result.call_date,
            audit_date=result.audit_date,
            duration_seconds=result.duration_seconds,
            agent_id=agent_id,
            agent_name=agent_name,
            customer_name=customer_name,
            customer_phone=customer_phone,
            transcription=result.transcription,
            language=result.language,
            call_summary=result.call_summary,
            customer_sentiment=result.customer_sentiment.model_dump(),
            agent_behavior=result.agent_behavior.model_dump(),
            compliance_risk=result.compliance_risk.model_dump(),
            question_scores=[q.model_dump() for q in result.question_scores],
            total_score=result.total_score,
            max_score=result.max_score,
            overall_percentage=result.overall_percentage,
            customer_intent=result.customer_intent,
            key_issues=result.key_issues,
            resolution_status=result.resolution_status,
            follow_up_required=result.follow_up_required,
            audio_file_path=file_path
        )
        db.add(db_record)
        db.commit()
        
        return result
        
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{call_id}", response_model=CallAnalysisResult)
async def get_call_analysis(call_id: str, db: Session = Depends(get_db)):
    """Get analysis results for a specific call"""
    record = db.query(CallAnalysis).filter(CallAnalysis.id == call_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    from ..models.schemas import CustomerSentiment, AgentBehavior, ComplianceRisk, QuestionScore
    
    return CallAnalysisResult(
        call_id=record.id,
        call_date=record.call_date,
        audit_date=record.audit_date,
        duration_seconds=record.duration_seconds,
        agent_id=record.agent_id,
        agent_name=record.agent_name,
        customer_name=record.customer_name,
        customer_phone=record.customer_phone,
        transcription=record.transcription,
        language=record.language,
        call_summary=record.call_summary,
        customer_sentiment=CustomerSentiment(**record.customer_sentiment),
        agent_behavior=AgentBehavior(**record.agent_behavior),
        compliance_risk=ComplianceRisk(**record.compliance_risk),
        question_scores=[QuestionScore(**q) for q in record.question_scores],
        total_score=record.total_score,
        max_score=record.max_score,
        overall_percentage=record.overall_percentage,
        customer_intent=record.customer_intent,
        key_issues=record.key_issues,
        resolution_status=record.resolution_status,
        follow_up_required=record.follow_up_required
    )


@router.get("/", response_model=List[dict])
async def list_calls(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all analyzed calls"""
    records = db.query(CallAnalysis).order_by(CallAnalysis.call_date.desc()).offset(skip).limit(limit).all()
    
    return [{
        "id": r.id,
        "agent_name": r.agent_name or "Unknown",
        "customer_name": r.customer_name or "Unknown",
        "call_date": r.call_date.isoformat(),
        "duration": r.duration_seconds,
        "overall_score": r.overall_percentage,
        "sentiment": r.customer_sentiment.get("overall_sentiment", "Unknown") if r.customer_sentiment else "Unknown",
        "resolution_status": r.resolution_status
    } for r in records]


@router.delete("/{call_id}")
async def delete_call(call_id: str, db: Session = Depends(get_db)):
    """Delete a call analysis record"""
    record = db.query(CallAnalysis).filter(CallAnalysis.id == call_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Delete audio file if exists
    if record.audio_file_path and os.path.exists(record.audio_file_path):
        os.remove(record.audio_file_path)
    
    db.delete(record)
    db.commit()
    
    return {"message": "Call deleted successfully"}

