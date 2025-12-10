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
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def _get_questionnaire(self) -> List[Dict[str, Any]]:
        """Return the questionnaire for call evaluation"""
        return [
            # Call Opening
            {"category": "Call Opening", "question": "Did agent probe customer name before continuing?", "max_score": 3},
            {"category": "Call Opening", "question": "Did agent open call as per timelines and script?", "max_score": 3},
            {"category": "Call Opening", "question": "Did agent give opening within 5 seconds?", "max_score": 2},
            {"category": "Call Opening", "question": "Did agent greet according to language selection?", "max_score": 2},
            
            # Soft Skills
            {"category": "Soft Skills", "question": "Did agent willingly help without making commitments?", "max_score": 3},
            {"category": "Soft Skills", "question": "Did agent use proper sentence structure and grammar?", "max_score": 3},
            {"category": "Soft Skills", "question": "Was agent confident during the call?", "max_score": 3},
            {"category": "Soft Skills", "question": "Did agent show empathy towards customer?", "max_score": 4},
            {"category": "Soft Skills", "question": "Did agent maintain professional tone throughout?", "max_score": 3},
            
            # Probing & Understanding
            {"category": "Probing & Understanding", "question": "Did agent ask effective questions to understand needs?", "max_score": 4},
            {"category": "Probing & Understanding", "question": "Did agent understand customer concern at first instance?", "max_score": 3},
            {"category": "Probing & Understanding", "question": "Did agent ask pertinent diagnostic questions?", "max_score": 3},
            
            # Problem Resolution
            {"category": "Problem Resolution", "question": "Did agent provide accurate information?", "max_score": 5},
            {"category": "Problem Resolution", "question": "Did agent offer appropriate solutions?", "max_score": 5},
            {"category": "Problem Resolution", "question": "Did agent handle objections effectively?", "max_score": 4},
            
            # Call Closing
            {"category": "Call Closing", "question": "Did agent follow correct closing format?", "max_score": 3},
            {"category": "Call Closing", "question": "Did agent summarize the call properly?", "max_score": 3},
            {"category": "Call Closing", "question": "Did agent ask for further assistance?", "max_score": 2},
            
            # Critical Parameters
            {"category": "Critical Parameters", "question": "Did agent NOT disconnect without warning?", "max_score": 10},
            {"category": "Critical Parameters", "question": "Did agent use correct categorization?", "max_score": 5},
        ]
    
    async def analyze_call(self, transcription: str) -> Dict[str, Any]:
        """
        Comprehensive call analysis using GPT-3.5 Turbo
        """
        # Get sentiment analysis
        sentiment_result = await self._analyze_sentiment(transcription)
        
        # Get agent behavior analysis
        agent_behavior = await self._analyze_agent_behavior(transcription)
        
        # Get compliance risk assessment
        compliance_risk = await self._assess_compliance_risk(transcription)
        
        # Score the call against questionnaire
        question_scores = await self._score_questionnaire(transcription)
        
        # Generate call summary
        call_summary = await self._generate_summary(transcription)
        
        # Extract customer intent and issues
        intent_analysis = await self._analyze_intent(transcription)
        
        return {
            "customer_sentiment": sentiment_result,
            "agent_behavior": agent_behavior,
            "compliance_risk": compliance_risk,
            "question_scores": question_scores,
            "call_summary": call_summary,
            "customer_intent": intent_analysis["intent"],
            "key_issues": intent_analysis["issues"],
            "resolution_status": intent_analysis["resolution_status"],
            "follow_up_required": intent_analysis["follow_up_required"]
        }
    
    async def _analyze_sentiment(self, transcription: str) -> CustomerSentiment:
        """Analyze customer sentiment from transcription"""
        prompt = f"""Analyze the customer sentiment from this call transcription and return a JSON object with these exact fields:
- overall_sentiment: one of "Positive", "Neutral", "Negative", "Mixed"
- emotions: array of emotions detected, each one of: "Calm", "Cooperative", "Confused", "Angry", "Frustrated", "Satisfied"
- urgency_level: one of "High", "Medium", "Low"
- frustration_indicator: boolean
- escalation_risk: number between 0-100 representing percentage
- call_opening_emotion: one of "Calm", "Cooperative", "Confused", "Angry", "Frustrated", "Satisfied"
- call_end_emotion: one of "Calm", "Cooperative", "Confused", "Angry", "Frustrated", "Satisfied"

Transcription:
{transcription}

Return ONLY valid JSON, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert for call center quality auditing. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return CustomerSentiment(
                overall_sentiment=SentimentType(result.get("overall_sentiment", "Neutral")),
                emotions=[EmotionType(e) for e in result.get("emotions", ["Calm"])],
                urgency_level=UrgencyLevel(result.get("urgency_level", "Medium")),
                frustration_indicator=result.get("frustration_indicator", False),
                escalation_risk=result.get("escalation_risk", 0),
                call_opening_emotion=EmotionType(result.get("call_opening_emotion", "Calm")),
                call_end_emotion=EmotionType(result.get("call_end_emotion", "Calm"))
            )
        except (json.JSONDecodeError, ValueError) as e:
            # Return default values if parsing fails
            return CustomerSentiment(
                overall_sentiment=SentimentType.NEUTRAL,
                emotions=[EmotionType.CALM],
                urgency_level=UrgencyLevel.MEDIUM,
                frustration_indicator=False,
                escalation_risk=0,
                call_opening_emotion=EmotionType.CALM,
                call_end_emotion=EmotionType.CALM
            )
    
    async def _analyze_agent_behavior(self, transcription: str) -> AgentBehavior:
        """Analyze agent behavior from transcription"""
        prompt = f"""Analyze the agent's behavior in this call transcription and return a JSON object with these exact boolean fields:
- calmness: was the agent calm throughout?
- confidence: did the agent sound confident?
- politeness: was the agent polite?
- empathy: did the agent show empathy?
- proper_grammar: did the agent use proper grammar?

Transcription:
{transcription}

Return ONLY valid JSON, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a call quality expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return AgentBehavior(
                calmness=result.get("calmness", True),
                confidence=result.get("confidence", True),
                politeness=result.get("politeness", True),
                empathy=result.get("empathy", True),
                proper_grammar=result.get("proper_grammar", True)
            )
        except json.JSONDecodeError:
            return AgentBehavior(
                calmness=True,
                confidence=True,
                politeness=True,
                empathy=True,
                proper_grammar=True
            )
    
    async def _assess_compliance_risk(self, transcription: str) -> ComplianceRisk:
        """Assess compliance risk from transcription"""
        prompt = f"""Assess the compliance risk in this call transcription and return a JSON object with:
- fraud_suspected: boolean indicating if fraud is suspected
- compliance_risk: one of "low", "medium", "high"
- trust_justification: brief explanation of the risk assessment

Transcription:
{transcription}

Return ONLY valid JSON, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a compliance risk expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return ComplianceRisk(
                fraud_suspected=result.get("fraud_suspected", False),
                compliance_risk=result.get("compliance_risk", "low"),
                trust_justification=result.get("trust_justification", "No concerns identified")
            )
        except json.JSONDecodeError:
            return ComplianceRisk(
                fraud_suspected=False,
                compliance_risk="low",
                trust_justification="Unable to assess"
            )
    
    async def _score_questionnaire(self, transcription: str) -> List[QuestionScore]:
        """Score the call against the questionnaire"""
        questionnaire = self._get_questionnaire()
        questions_text = "\n".join([f"{i+1}. [{q['category']}] {q['question']} (max: {q['max_score']} points)" 
                                    for i, q in enumerate(questionnaire)])
        
        prompt = f"""Score this call against each question. For each question, provide:
- score: points earned (0 to max_score)
- answer: brief explanation (Yes/No/NA with reason)

Questions:
{questions_text}

Transcription:
{transcription}

Return a JSON array with objects containing: category, question, answer, score, max_score
Return ONLY valid JSON array, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a call quality auditor. Score calls fairly based on the evidence in the transcription. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            results = json.loads(response.choices[0].message.content)
            return [QuestionScore(
                category=r.get("category", "Unknown"),
                question=r.get("question", ""),
                answer=r.get("answer", "NA"),
                score=min(r.get("score", 0), r.get("max_score", 5)),
                max_score=r.get("max_score", 5)
            ) for r in results]
        except json.JSONDecodeError:
            # Return default scores if parsing fails
            return [QuestionScore(
                category=q["category"],
                question=q["question"],
                answer="Unable to assess",
                score=0,
                max_score=q["max_score"]
            ) for q in questionnaire]
    
    async def _generate_summary(self, transcription: str) -> str:
        """Generate a summary of the call"""
        prompt = f"""Provide a brief 2-3 sentence summary of this customer service call:

{transcription}

Focus on: the customer's issue, what action was taken, and the outcome."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a call summarization expert. Be concise and factual."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    async def _analyze_intent(self, transcription: str) -> Dict[str, Any]:
        """Analyze customer intent and key issues"""
        prompt = f"""Analyze this call and return a JSON object with:
- intent: customer's primary intent (e.g., "Complaint", "Query", "Feedback", "Request")
- issues: array of specific issues raised
- resolution_status: one of "Resolved", "Partially Resolved", "Unresolved", "Requires Follow-up"
- follow_up_required: boolean

Transcription:
{transcription}

Return ONLY valid JSON, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a customer intent analysis expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "intent": "Unknown",
                "issues": [],
                "resolution_status": "Unknown",
                "follow_up_required": False
            }


sentiment_service = SentimentAnalysisService()

