from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict

from ..models.database import get_db, CallAnalysis
from ..models.schemas import DashboardMetrics

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(db: Session = Depends(get_db)):
    """Get aggregated metrics for the dashboard"""
    
    # Get all records
    records = db.query(CallAnalysis).all()
    
    if not records:
        return DashboardMetrics(
            total_calls=0,
            avg_score=0,
            sentiment_distribution={"Positive": 0, "Neutral": 0, "Negative": 0, "Mixed": 0},
            urgency_distribution={"High": 0, "Medium": 0, "Low": 0},
            escalation_rate=0,
            avg_call_duration=0,
            top_issues=[],
            agent_performance=[],
            daily_trends=[]
        )
    
    # Calculate metrics
    total_calls = len(records)
    avg_score = sum(r.overall_percentage for r in records) / total_calls
    avg_duration = sum(r.duration_seconds or 0 for r in records) / total_calls
    
    # Sentiment distribution
    sentiment_dist = defaultdict(int)
    urgency_dist = defaultdict(int)
    escalation_risks = []
    
    for r in records:
        if r.customer_sentiment:
            sentiment = r.customer_sentiment.get("overall_sentiment", "Unknown")
            sentiment_dist[sentiment] += 1
            
            urgency = r.customer_sentiment.get("urgency_level", "Unknown")
            urgency_dist[urgency] += 1
            
            escalation_risks.append(r.customer_sentiment.get("escalation_risk", 0))
    
    # Calculate escalation rate (calls with >50% escalation risk)
    high_escalation = sum(1 for risk in escalation_risks if risk > 50)
    escalation_rate = (high_escalation / total_calls * 100) if total_calls > 0 else 0
    
    # Top issues
    all_issues = []
    for r in records:
        if r.key_issues:
            all_issues.extend(r.key_issues)
    
    issue_counts = defaultdict(int)
    for issue in all_issues:
        issue_counts[issue] += 1
    
    top_issues = [{"issue": k, "count": v} for k, v in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]]
    
    # Agent performance
    agent_scores = defaultdict(list)
    for r in records:
        agent_name = r.agent_name or "Unknown"
        agent_scores[agent_name].append(r.overall_percentage)
    
    agent_performance = [
        {"agent": name, "avg_score": sum(scores)/len(scores), "total_calls": len(scores)}
        for name, scores in agent_scores.items()
    ]
    agent_performance.sort(key=lambda x: -x["avg_score"])
    
    # Daily trends (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_data = defaultdict(lambda: {"calls": 0, "total_score": 0, "positive": 0, "negative": 0})
    
    for r in records:
        if r.call_date and r.call_date >= thirty_days_ago:
            day = r.call_date.strftime("%Y-%m-%d")
            daily_data[day]["calls"] += 1
            daily_data[day]["total_score"] += r.overall_percentage
            
            if r.customer_sentiment:
                sentiment = r.customer_sentiment.get("overall_sentiment", "")
                if sentiment == "Positive":
                    daily_data[day]["positive"] += 1
                elif sentiment == "Negative":
                    daily_data[day]["negative"] += 1
    
    daily_trends = [
        {
            "date": date,
            "calls": data["calls"],
            "avg_score": data["total_score"] / data["calls"] if data["calls"] > 0 else 0,
            "positive_calls": data["positive"],
            "negative_calls": data["negative"]
        }
        for date, data in sorted(daily_data.items())
    ]
    
    return DashboardMetrics(
        total_calls=total_calls,
        avg_score=round(avg_score, 2),
        sentiment_distribution=dict(sentiment_dist),
        urgency_distribution=dict(urgency_dist),
        escalation_rate=round(escalation_rate, 2),
        avg_call_duration=round(avg_duration, 2),
        top_issues=top_issues,
        agent_performance=agent_performance[:10],
        daily_trends=daily_trends
    )


@router.get("/charts/sentiment-pie")
async def get_sentiment_pie_chart(db: Session = Depends(get_db)):
    """Get sentiment distribution data for pie chart"""
    records = db.query(CallAnalysis).all()
    
    sentiment_counts = defaultdict(int)
    for r in records:
        if r.customer_sentiment:
            sentiment = r.customer_sentiment.get("overall_sentiment", "Unknown")
            sentiment_counts[sentiment] += 1
    
    return {
        "labels": list(sentiment_counts.keys()),
        "values": list(sentiment_counts.values()),
        "colors": {
            "Positive": "#10B981",
            "Neutral": "#6B7280",
            "Negative": "#EF4444",
            "Mixed": "#F59E0B"
        }
    }


@router.get("/charts/agent-performance")
async def get_agent_performance_chart(db: Session = Depends(get_db)):
    """Get agent performance data for bar chart"""
    records = db.query(CallAnalysis).all()
    
    agent_data = defaultdict(lambda: {"scores": [], "sentiments": defaultdict(int)})
    
    for r in records:
        agent = r.agent_name or "Unknown"
        agent_data[agent]["scores"].append(r.overall_percentage)
        if r.customer_sentiment:
            sentiment = r.customer_sentiment.get("overall_sentiment", "Unknown")
            agent_data[agent]["sentiments"][sentiment] += 1
    
    result = []
    for agent, data in agent_data.items():
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        result.append({
            "agent": agent,
            "avg_score": round(avg_score, 2),
            "total_calls": len(data["scores"]),
            "positive_calls": data["sentiments"].get("Positive", 0),
            "negative_calls": data["sentiments"].get("Negative", 0)
        })
    
    return sorted(result, key=lambda x: -x["avg_score"])


@router.get("/charts/daily-trends")
async def get_daily_trends_chart(db: Session = Depends(get_db)):
    """Get daily call trends for line chart"""
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    records = db.query(CallAnalysis).filter(CallAnalysis.call_date >= thirty_days_ago).all()
    
    daily_data = defaultdict(lambda: {
        "calls": 0, "total_score": 0, "positive": 0, "negative": 0, "neutral": 0
    })
    
    for r in records:
        if r.call_date:
            day = r.call_date.strftime("%Y-%m-%d")
            daily_data[day]["calls"] += 1
            daily_data[day]["total_score"] += r.overall_percentage
            
            if r.customer_sentiment:
                sentiment = r.customer_sentiment.get("overall_sentiment", "Neutral")
                if sentiment == "Positive":
                    daily_data[day]["positive"] += 1
                elif sentiment == "Negative":
                    daily_data[day]["negative"] += 1
                else:
                    daily_data[day]["neutral"] += 1
    
    dates = sorted(daily_data.keys())
    
    return {
        "dates": dates,
        "calls": [daily_data[d]["calls"] for d in dates],
        "avg_scores": [daily_data[d]["total_score"] / daily_data[d]["calls"] if daily_data[d]["calls"] > 0 else 0 for d in dates],
        "positive": [daily_data[d]["positive"] for d in dates],
        "negative": [daily_data[d]["negative"] for d in dates],
        "neutral": [daily_data[d]["neutral"] for d in dates]
    }


@router.get("/charts/category-scores")
async def get_category_scores_chart(db: Session = Depends(get_db)):
    """Get average scores by category for radar/bar chart"""
    records = db.query(CallAnalysis).all()
    
    category_data = defaultdict(lambda: {"total_score": 0, "max_score": 0, "count": 0})
    
    for r in records:
        if r.question_scores:
            for q in r.question_scores:
                cat = q.get("category", "Unknown")
                category_data[cat]["total_score"] += q.get("score", 0)
                category_data[cat]["max_score"] += q.get("max_score", 0)
                category_data[cat]["count"] += 1
    
    result = []
    for category, data in category_data.items():
        percentage = (data["total_score"] / data["max_score"] * 100) if data["max_score"] > 0 else 0
        result.append({
            "category": category,
            "avg_percentage": round(percentage, 2),
            "total_score": data["total_score"],
            "max_score": data["max_score"]
        })
    
    return sorted(result, key=lambda x: -x["avg_percentage"])


@router.get("/charts/urgency-distribution")
async def get_urgency_distribution_chart(db: Session = Depends(get_db)):
    """Get urgency level distribution for donut chart"""
    records = db.query(CallAnalysis).all()
    
    urgency_counts = defaultdict(int)
    for r in records:
        if r.customer_sentiment:
            urgency = r.customer_sentiment.get("urgency_level", "Unknown")
            urgency_counts[urgency] += 1
    
    return {
        "labels": list(urgency_counts.keys()),
        "values": list(urgency_counts.values()),
        "colors": {
            "High": "#EF4444",
            "Medium": "#F59E0B",
            "Low": "#10B981"
        }
    }


@router.get("/charts/escalation-risk")
async def get_escalation_risk_chart(db: Session = Depends(get_db)):
    """Get escalation risk distribution for histogram"""
    records = db.query(CallAnalysis).all()
    
    risk_buckets = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
    
    for r in records:
        if r.customer_sentiment:
            risk = r.customer_sentiment.get("escalation_risk", 0)
            if risk <= 20:
                risk_buckets["0-20%"] += 1
            elif risk <= 40:
                risk_buckets["20-40%"] += 1
            elif risk <= 60:
                risk_buckets["40-60%"] += 1
            elif risk <= 80:
                risk_buckets["60-80%"] += 1
            else:
                risk_buckets["80-100%"] += 1
    
    return {
        "labels": list(risk_buckets.keys()),
        "values": list(risk_buckets.values()),
        "colors": ["#10B981", "#84CC16", "#F59E0B", "#F97316", "#EF4444"]
    }

