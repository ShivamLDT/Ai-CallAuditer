import os
import json
from openai import OpenAI
from typing import Dict, Any, List
from ..models.schemas import (
    CustomerSentiment, AgentBehavior, ComplianceRisk, QuestionScore,
    SentimentType, UrgencyLevel, EmotionType
)


class SentimentAnalysisService:
    def __init__(self):
        self._client = None
        self.model = "gpt-3.5-turbo"
    
    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def _get_questionnaire(self) -> List[Dict[str, Any]]:
        """Return the questionnaire for call evaluation"""
        return [
            {"category": "Call Opening", "question": "Did agent probe customer name before continuing?", "max_score": 3},
            {"category": "Call Opening", "question": "Did agent open call as per timelines and script?", "max_score": 3},
            {"category": "Call Opening", "question": "Did agent give opening within 5 seconds?", "max_score": 2},
            {"category": "Call Opening", "question": "Did agent greet according to language selection?", "max_score": 2},
            {"category": "Soft Skills", "question": "Did agent willingly help without making commitments?", "max_score": 3},
            {"category": "Soft Skills", "question": "Did agent use proper sentence structure and grammar?", "max_score": 3},
            {"category": "Soft Skills", "question": "Was agent confident during the call?", "max_score": 3},
            {"category": "Soft Skills", "question": "Did agent show empathy towards customer?", "max_score": 4},
            {"category": "Soft Skills", "question": "Did agent maintain professional tone throughout?", "max_score": 3},
            {"category": "Probing & Understanding", "question": "Did agent ask effective questions to understand needs?", "max_score": 4},
            {"category": "Probing & Understanding", "question": "Did agent understand customer concern at first instance?", "max_score": 3},
            {"category": "Probing & Understanding", "question": "Did agent ask pertinent diagnostic questions?", "max_score": 3},
            {"category": "Problem Resolution", "question": "Did agent provide accurate information?", "max_score": 5},
            {"category": "Problem Resolution", "question": "Did agent offer appropriate solutions?", "max_score": 5},
            {"category": "Problem Resolution", "question": "Did agent handle objections effectively?", "max_score": 4},
            {"category": "Call Closing", "question": "Did agent follow correct closing format?", "max_score": 3},
            {"category": "Call Closing", "question": "Did agent summarize the call properly?", "max_score": 3},
            {"category": "Call Closing", "question": "Did agent ask for further assistance?", "max_score": 2},
            {"category": "Critical Parameters", "question": "Did agent NOT disconnect without warning?", "max_score": 10},
            {"category": "Critical Parameters", "question": "Did agent use correct categorization?", "max_score": 5},
        ]
    
    async def analyze_call(self, transcription: str) -> Dict[str, Any]:
        """
        OPTIMIZED: Single comprehensive API call for all analysis
        Reduces 6 API calls to 1, saving ~10-15 seconds
        """
        questionnaire = self._get_questionnaire()
        questions_text = "\n".join([
            f"{i+1}. [{q['category']}] {q['question']} (max: {q['max_score']} points)" 
            for i, q in enumerate(questionnaire)
        ])
        
        prompt = f"""Analyze this customer service call transcription comprehensively. Return a single JSON object with ALL of the following sections:

1. "customer_sentiment": {{
   "overall_sentiment": "Positive" | "Neutral" | "Negative" | "Mixed",
   "emotions": ["Calm", "Frustrated", etc.],
   "urgency_level": "High" | "Medium" | "Low",
   "frustration_indicator": true/false,
   "escalation_risk": 0-100,
   "call_opening_emotion": "Calm" | "Frustrated" | etc.,
   "call_end_emotion": "Calm" | "Satisfied" | etc.
}}

2. "agent_behavior": {{
   "calmness": true/false,
   "confidence": true/false,
   "politeness": true/false,
   "empathy": true/false,
   "proper_grammar": true/false
}}

3. "compliance_risk": {{
   "fraud_suspected": true/false,
   "compliance_risk": "low" | "medium" | "high",
   "trust_justification": "brief explanation"
}}

4. "call_summary": "2-3 sentence summary of the call"

5. "customer_intent": "Complaint" | "Query" | "Feedback" | "Request"

6. "key_issues": ["issue1", "issue2"]

7. "resolution_status": "Resolved" | "Partially Resolved" | "Unresolved" | "Requires Follow-up"

8. "follow_up_required": true/false

9. "question_scores": [
   {{"category": "...", "question": "...", "answer": "Yes/No/Unable to assess", "score": 0-max, "max_score": max}}
]

Questions to score:
{questions_text}

TRANSCRIPTION:
{transcription}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation. Ensure all fields are present."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert call quality auditor. Analyze calls thoroughly and return comprehensive JSON analysis. Always respond with valid JSON only, no markdown formatting."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2500
        )
        
        try:
            # Clean response - remove any markdown formatting
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            result = json.loads(content)
            
            # Parse customer sentiment
            sentiment_data = result.get("customer_sentiment", {})
            customer_sentiment = CustomerSentiment(
                overall_sentiment=SentimentType(sentiment_data.get("overall_sentiment", "Neutral")),
                emotions=[EmotionType(e) for e in sentiment_data.get("emotions", ["Calm"])[:5]],
                urgency_level=UrgencyLevel(sentiment_data.get("urgency_level", "Medium")),
                frustration_indicator=sentiment_data.get("frustration_indicator", False),
                escalation_risk=min(100, max(0, sentiment_data.get("escalation_risk", 0))),
                call_opening_emotion=EmotionType(sentiment_data.get("call_opening_emotion", "Calm")),
                call_end_emotion=EmotionType(sentiment_data.get("call_end_emotion", "Calm"))
            )
            
            # Parse agent behavior
            behavior_data = result.get("agent_behavior", {})
            agent_behavior = AgentBehavior(
                calmness=behavior_data.get("calmness", True),
                confidence=behavior_data.get("confidence", True),
                politeness=behavior_data.get("politeness", True),
                empathy=behavior_data.get("empathy", True),
                proper_grammar=behavior_data.get("proper_grammar", True)
            )
            
            # Parse compliance risk
            compliance_data = result.get("compliance_risk", {})
            compliance_risk = ComplianceRisk(
                fraud_suspected=compliance_data.get("fraud_suspected", False),
                compliance_risk=compliance_data.get("compliance_risk", "low"),
                trust_justification=compliance_data.get("trust_justification", "No concerns identified")
            )
            
            # Parse question scores
            scores_data = result.get("question_scores", [])
            if scores_data:
                question_scores = [
                    QuestionScore(
                        category=s.get("category", "Unknown"),
                        question=s.get("question", ""),
                        answer=s.get("answer", "Unable to assess"),
                        score=min(s.get("score", 0), s.get("max_score", 5)),
                        max_score=s.get("max_score", 5)
                    ) for s in scores_data
                ]
            else:
                # Fallback to default scores
                question_scores = [
                    QuestionScore(
                        category=q["category"],
                        question=q["question"],
                        answer="Unable to assess",
                        score=0,
                        max_score=q["max_score"]
                    ) for q in questionnaire
                ]
            
            return {
                "customer_sentiment": customer_sentiment,
                "agent_behavior": agent_behavior,
                "compliance_risk": compliance_risk,
                "question_scores": question_scores,
                "call_summary": result.get("call_summary", "Summary not available"),
                "customer_intent": result.get("customer_intent", "Unknown"),
                "key_issues": result.get("key_issues", []),
                "resolution_status": result.get("resolution_status", "Unknown"),
                "follow_up_required": result.get("follow_up_required", False)
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Analysis parsing error: {e}")
            # Return defaults if parsing fails
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when parsing fails"""
        questionnaire = self._get_questionnaire()
        return {
            "customer_sentiment": CustomerSentiment(
                overall_sentiment=SentimentType.NEUTRAL,
                emotions=[EmotionType.CALM],
                urgency_level=UrgencyLevel.MEDIUM,
                frustration_indicator=False,
                escalation_risk=0,
                call_opening_emotion=EmotionType.CALM,
                call_end_emotion=EmotionType.CALM
            ),
            "agent_behavior": AgentBehavior(
                calmness=True,
                confidence=True,
                politeness=True,
                empathy=True,
                proper_grammar=True
            ),
            "compliance_risk": ComplianceRisk(
                fraud_suspected=False,
                compliance_risk="low",
                trust_justification="Unable to assess"
            ),
            "question_scores": [
                QuestionScore(
                    category=q["category"],
                    question=q["question"],
                    answer="Unable to assess",
                    score=0,
                    max_score=q["max_score"]
                ) for q in questionnaire
            ],
            "call_summary": "Analysis could not be completed",
            "customer_intent": "Unknown",
            "key_issues": [],
            "resolution_status": "Unknown",
            "follow_up_required": False
        }


sentiment_service = SentimentAnalysisService()
