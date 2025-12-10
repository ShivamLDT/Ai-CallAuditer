from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./call_auditor.db")

# Configure engine based on database type
if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL (Supabase) configuration
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
else:
    # SQLite configuration (for local development)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class CallAnalysis(Base):
    __tablename__ = "call_analyses"

    id = Column(String, primary_key=True, index=True)
    call_date = Column(DateTime, default=datetime.utcnow)
    audit_date = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float)
    
    # Agent Info
    agent_id = Column(String, nullable=True)
    agent_name = Column(String, nullable=True)
    
    # Customer Info
    customer_name = Column(String, nullable=True)
    customer_phone = Column(String, nullable=True)
    
    # Transcription
    transcription = Column(Text)
    language = Column(String)
    
    # Analysis Results (stored as JSON)
    call_summary = Column(Text)
    customer_sentiment = Column(JSON)
    agent_behavior = Column(JSON)
    compliance_risk = Column(JSON)
    question_scores = Column(JSON)
    
    # Scoring
    total_score = Column(Integer)
    max_score = Column(Integer)
    overall_percentage = Column(Float)
    
    # Intent & Insights
    customer_intent = Column(String)
    key_issues = Column(JSON)
    resolution_status = Column(String)
    follow_up_required = Column(Boolean)
    
    # Audio Storage (Supabase Storage)
    audio_file_path = Column(String, nullable=True)  # Legacy: local file path (deprecated)
    audio_storage_path = Column(String, nullable=True)  # Supabase Storage path
    recording_expires_at = Column(DateTime, nullable=True)  # Auto-deletion date


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
