import os
from datetime import datetime
import uuid
import numpy as np
import json
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.logging import logger, logging
from src.config import config

from src.utils import classify_emotion, classify_intent, embed_semantic, embed_emotion, cosine_similarity_score, screen_safety
from src.core.profile_manager import ProfileManager
from src.core.vector_store import vector_store # Assuming vector_store is an initialized instance

# Initialize Profile Manager
profile_manager = ProfileManager(profile_dir=config.get("general.profile_dir"))

# Initialize Groq LLM
llm = ChatGroq(
    temperature=config.get("llm.temperature"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=config.get("llm.model_name")
)

class Chatbot:
    def __init__(self):
        self.message_history = {} # Stores recent messages for context

    def _load_recent_messages(self, user_id: str, k: int = 12) -> List[Dict[str, Any]]:
        """Loads recent messages for a user."""
        return self.message_history.get(user_id, [])[-k:]

    def _log_turn(self, user_id: str, user_text: str, emo: dict, intent: str, context: List[Dict[str, Any]], assistant_response: str):
        """Logs the user and assistant turn."""
        if user_id not in self.message_history:
            self.message_history[user_id] = []

        user_message = {
            "msg_id": str(uuid.uuid4()),
            "user_id": user_id,
            "role": "user",
            "text": user_text,
            "timestamp": datetime.now().isoformat(),
            "emotion": emo,
            "intent": intent,
            "embeddings": {
                "semantic": embed_semantic(user_text),
                "emotion": embed_emotion(emo)
            },
            "safety_flags": screen_safety(user_text, emo) # Re-screen for logging
        }
        self.message_history[user_id].append(user_message)

        assistant_message = {
            "msg_id": str(uuid.uuid4()),
            "user_id": user_id,
            "role": "assistant",
            "text": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "context_chunks": [c["doc_id"] for c in context] if context else [],
            # Emotion and intent for assistant response could be inferred or set to neutral
            "emotion": {"primary": "neutral", "scores": {"neutral": 1.0}},
            "intent": "respond",
            "safety_flags": {"self_harm": 0.0} # Assuming LLM output is safe after refinement
        }
        self.message_history[user_id].append(assistant_message)
        # Flush file handler to ensure immediate writing to log file
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

    def _system_rules(self) -> str:
        """Defines the system rules for the chatbot."""
        return """You are a mental wellbeing chatbot designed to provide supportive, personalized, context-aware, and emotion-aware conversations.
        Your goal is to help users manage their mental wellbeing.
        Always respond in a compassionate, empathetic, and professional manner.
        Avoid conversational embellishments like "*warm smile*" or similar non-verbal cues.
        Prioritize user safety.
        """

    def _score_chunk(self, q_embed_sem: List[float], q_embed_em: List[float], p_tags: List[str], chunk: Dict[str, Any], profile: Dict[str, Any]) -> float:
        """Scores a retrieved chunk based on semantic, emotion, profile tags, recency, and style match."""
        w_sem = 0.55
        w_em = 0.2
        w_prof = 0.1
        w_rec = 0.1
        w_sft = 0.05

        # Semantic similarity
        s_sem = cosine_similarity_score(q_embed_sem, chunk["embeddings"]["semantic"])

        # Emotion similarity (assuming chunks have emotion embeddings or a neutral default)
        # For KB chunks, we used a dummy neutral emotion embedding.
        # A more sophisticated system would pre-classify emotions for KB chunks.
        # For now, this will mostly match if the query emotion is "neutral".
        s_em = cosine_similarity_score(q_embed_em, chunk["embeddings"]["emotion"])

        # Profile tag Jaccard similarity
        chunk_meta_tags = []
        if chunk["meta"].get("therapy"):
            chunk_meta_tags.append(chunk["meta"]["therapy"])
        if chunk["meta"].get("topic"):
            chunk_meta_tags.append(chunk["meta"]["topic"])
        if chunk["meta"].get("style"):
            chunk_meta_tags.append(chunk["meta"]["style"])
        
        intersection = len(set(p_tags).intersection(set(chunk_meta_tags)))
        union = len(set(p_tags).union(set(chunk_meta_tags)))
        s_prof = intersection / union if union > 0 else 0.0

        # Recency boost
        chunk_updated_at = datetime.fromisoformat(chunk["updated_at"])
        delta_days = (datetime.now() - chunk_updated_at).days
        s_rec = np.exp(-delta_days / 90)

        # Style match (simplified)
        user_tone_preference = profile.get("preferences", {}).get("tone", "genuine")
        chunk_style = chunk["meta"].get("style", "neutral") # Assuming chunks have a style meta
        s_sft = 1.0 if user_tone_preference.lower() == chunk_style.lower() else 0.0

        score = (w_sem * s_sem +
                 w_em * s_em +
                 w_prof * s_prof +
                 w_rec * s_rec +
                 w_sft * s_sft)
        return score

    def _assemble_context(self, rescored_chunks: List[tuple], prior_messages: List[Dict[str, Any]], profile: Dict[str, Any]) -> str:
        """Assembles the context window for the LLM prompt."""
        context_parts = []

        # 2-3 KB chunks (max 800-1200 tokens)
        # Sort by score and take top 3
        sorted_chunks = sorted(rescored_chunks, key=lambda x: x[0], reverse=True)[:3]
        for score, chunk in sorted_chunks:
            context_parts.append(f"Knowledge Base Chunk (Score: {score:.2f}, Source: {chunk.get('source')}): {chunk['text']}")

        # 2 recent user turns exhibiting current emotion (simplified to just recent turns for now)
        # In a real system, you'd filter for emotion. For now, just take the last 2 user messages.
        recent_user_messages = [msg for msg in prior_messages if msg["role"] == "user"][-2:]
        for msg in recent_user_messages:
            context_parts.append(f"Recent User Message (Emotion: {msg['emotion']['primary']}): {msg['text']}")

        # Profile summary (one-line)
        profile_summary = f"User Profile Summary: Age: {profile.get('demographics', {}).get('age', 'N/A')}, Conditions: {', '.join(profile.get('conditions', []))}, Goals: {', '.join(profile.get('goals', []))}, Preferred Styles: {', '.join(profile.get('therapeutic_styles', []))}, Tone: {profile.get('preferences', {}).get('tone', 'N/A')}."
        context_parts.append(profile_summary)

        # Safety plan snippet if relevant (simplified)
        if profile.get("safety_plan", {}).get("grounding_preference"):
            context_parts.append(f"Safety Plan Snippet: Grounding preference: {profile['safety_plan']['grounding_preference']}")

        return "\n\n".join(context_parts)

    def _build_prompt(self) -> ChatPromptTemplate:
        """Builds the prompt for the LLM."""
        template = ChatPromptTemplate.from_messages([
            ("system", "{system_rules}"),
            ("system", "User Profile: {profile_str}"),
            ("system", "Detected User Emotion: {emotion_primary} (Scores: {emotion_scores_str})"),
            ("system", "Detected User Intent: {intent}"),
            ("system", "Context for response:\n{context_str}"),
            ("human", "{user_text}")
        ])
        return template

    def _get_prompt_inputs(self, system_rules: str, profile: Dict[str, Any], emo: dict, intent: str, context_str: str, user_text: str) -> Dict[str, Any]:
        """Prepares the input dictionary for the prompt template."""
        profile_str = json.dumps(profile) # Convert profile dict to string
        emotion_scores_str = json.dumps(emo['scores']) # Convert emotion scores dict to string
        return {
            "system_rules": system_rules,
            "profile_str": profile_str,
            "emotion_primary": emo['primary'],
            "emotion_scores_str": emotion_scores_str,
            "intent": intent,
            "context_str": context_str,
            "user_text": user_text
        }

    def _safety_refine(self, draft: str, safety_flags: Dict[str, Any], profile: Dict[str, Any]) -> str:
        """Refines the LLM draft based on safety flags."""
        # This is a placeholder for a more robust safety refinement mechanism.
        # Could involve another LLM call or rule-based filtering.
        if safety_flags["level"] == "crisis":
            return "It sounds like you're going through a lot right now. Please reach out to a crisis hotline or a mental health professional immediately. I'm here to listen, but I'm not a substitute for professional help."
        
        # Check for avoid topics
        avoid_topics = profile.get("preferences", {}).get("avoid_topics", [])
        for topic in avoid_topics:
            if topic.lower() in draft.lower():
                return "I've noticed we might be touching on a sensitive topic. Would you like to steer our conversation in a different direction?"
        
        return draft

    def _maybe_update_profile(self, user_id: str, user_text: str, assistant_response: str, intent: str, emo: Dict[str, Any], rescored_chunks: List[tuple]):
        """Updates the user profile based on the conversation."""
        current_profile = profile_manager.load_profile(user_id)
        profile_patch = {}
        profile_updated_this_turn = False

        # Option 1: Emotion-Driven Therapeutic Style Preference
        positive_emotions = ["joy", "hopeful"]
        if emo["primary"] in positive_emotions:
            updated_styles = set(current_profile.get("therapeutic_styles", []))
            
            for score, chunk in rescored_chunks:
                if chunk["meta"].get("therapy") and chunk["meta"]["therapy"] not in updated_styles:
                    updated_styles.add(chunk["meta"]["therapy"])
                    logger.debug(f"Profile Update (Option 1): Added therapeutic style '{chunk['meta']['therapy']}' for user {user_id} due to positive emotion.")
            
            if updated_styles != set(current_profile.get("therapeutic_styles", [])):
                profile_patch["therapeutic_styles"] = list(updated_styles)
                profile_updated_this_turn = True
                logger.info(f"User {user_id} profile will be updated with new therapeutic styles: {list(updated_styles)}")

        class GoalConditionExtraction(BaseModel):
            goals: List[str] = Field(default_factory=list, description="List of mental wellbeing goals mentioned by the user.")
            conditions: List[str] = Field(default_factory=list, description="List of mental wellbeing conditions mentioned by the user.")

        # Option 2: LLM-Based Goal/Condition Extraction
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in extracting mental wellbeing goals and conditions from conversation.
            Analyze the user's message and the assistant's response to identify any new, explicit mental wellbeing goals or conditions mentioned by the user.
            """),
            ("human", "User message: {user_text}\nAssistant response: {assistant_response}")
        ])
        extraction_chain = extraction_prompt | llm.with_structured_output(GoalConditionExtraction)

        try:
            extraction_result: GoalConditionExtraction = extraction_chain.invoke({"user_text": user_text, "assistant_response": assistant_response})
            
            new_goals = extraction_result.goals
            new_conditions = extraction_result.conditions

            # Update goals
            current_goals = set(current_profile.get("goals", []))
            for goal in new_goals:
                if goal not in current_goals:
                    current_goals.add(goal)
                    profile_updated_this_turn = True
                    logger.debug(f"Profile Update (Option 2): Added new goal '{goal}' for user {user_id}.")
            
            # Update conditions
            current_conditions = set(current_profile.get("conditions", []))
            for condition in new_conditions:
                if condition not in current_conditions:
                    current_conditions.add(condition)
                    profile_updated_this_turn = True
                    logger.debug(f"Profile Update (Option 2): Added new condition '{condition}' for user {user_id}.")
            
            # Merge goals and conditions into the patch
            if "goals" in profile_patch:
                profile_patch["goals"].extend(list(current_goals - set(profile_patch["goals"])))
            else:
                profile_patch["goals"] = list(current_goals)

            if "conditions" in profile_patch:
                profile_patch["conditions"].extend(list(current_conditions - set(profile_patch["conditions"])))
            else:
                profile_patch["conditions"] = list(current_conditions)

            if profile_updated_this_turn:
                logger.info(f"User {user_id} profile will be updated with new goals/conditions.")

        except Exception as e:
            logger.error(f"Error during LLM-based goal/condition extraction: {e}")

        class AvoidTopicsExtraction(BaseModel):
            avoid_topics: List[str] = Field(default_factory=list, description="List of sensitive topics identified from the user's message that might be causing distress.")

        # Option 3: Negative Emotion-Driven "Avoid Topics" Update
        negative_emotions = ["sadness", "anger", "fear", "disgust", "anxious"]
        if emo["primary"] in negative_emotions:
            topic_extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert in identifying sensitive topics from user messages.
                Analyze the user's message and identify any potential sensitive topics that might be causing distress.
                """),
                ("human", "User message: {user_text}")
            ])
            topic_extraction_chain = topic_extraction_prompt | llm.with_structured_output(AvoidTopicsExtraction)

            try:
                topic_result: AvoidTopicsExtraction = topic_extraction_chain.invoke({"user_text": user_text})
                
                new_avoid_topics = topic_result.avoid_topics

                current_avoid_topics = set(current_profile.get("preferences", {}).get("avoid_topics", []))
                for topic in new_avoid_topics:
                    if topic not in current_avoid_topics:
                        current_avoid_topics.add(topic)
                        profile_updated_this_turn = True
                        logger.debug(f"Profile Update (Option 3): Added new avoid topic '{topic}' for user {user_id} due to negative emotion.")
                
                if profile_updated_this_turn:
                    updated_preferences = current_profile.get("preferences", {}).copy()
                    updated_preferences["avoid_topics"] = list(current_avoid_topics)
                    profile_patch["preferences"] = updated_preferences
                    logger.info(f"User {user_id} profile will be updated with new avoid topics: {list(current_avoid_topics)}")

            except Exception as e:
                logger.error(f"Error during LLM-based avoid topics extraction: {e}")
        
        # Apply the aggregated patch if any updates occurred
        if profile_updated_this_turn:
            profile_manager.upsert_profile(user_id, profile_patch)
            logger.info(f"User {user_id} profile updated with aggregated changes: {profile_patch}")

    def handle_turn(self, user_id: str, user_text: str) -> str:
        """Handles a single turn of the chatbot conversation."""
        profile = profile_manager.load_profile(user_id)
        prior_messages = self._load_recent_messages(user_id, k=12)

        emo = classify_emotion(user_text)
        logger.debug(f"Detected User Emotion: {emo['primary']} (Scores: {json.dumps(emo['scores'])})")
        intent = classify_intent(user_text)
        safety = screen_safety(user_text, emo)

        if safety["level"] == "crisis":
            final_response = self._safety_refine("", safety, profile) # Pass empty draft for crisis
            self._log_turn(user_id, user_text, emo, intent, [], final_response)
            return final_response

        q_embed_sem = embed_semantic(user_text)
        q_embed_em = embed_emotion(emo)
        p_tags = profile_manager.profile_tags(profile)
        
        filters = profile_manager.profile_filters(profile)
        cands = vector_store.vector_search(q_embed_sem, q_embed_em, filters=filters, top_k=50)

        rescored = []
        for c in cands:
            # Assuming KB chunks have pre-computed semantic and emotion embeddings
            # For simplicity, we'll add dummy embeddings if they don't exist for now.
            if "embeddings" not in c:
                c["embeddings"] = {
                    "semantic": embed_semantic(c["text"]),
                    "emotion": embed_emotion({"primary": "neutral", "scores": {"neutral": 1.0}})
                }
            score = self._score_chunk(q_embed_sem, q_embed_em, p_tags, c, profile)
            rescored.append((score, c))
        
        context_str = self._assemble_context(rescored, prior_messages, profile)
        prompt_template = self._build_prompt()
        prompt_inputs = self._get_prompt_inputs(self._system_rules(), profile, emo, intent, context_str, user_text)
        
        # Removed verbose logging of prompt inputs to improve log readability
        # logger.debug(f"Prompt Inputs: {json.dumps(prompt_inputs, indent=2)}") 
        chain = prompt_template | llm | StrOutputParser()
        draft = chain.invoke(prompt_inputs)

        final_response = self._safety_refine(draft, safety, profile)
        self._log_turn(user_id, user_text, emo, intent, [c for s, c in rescored], final_response)
        self._maybe_update_profile(user_id, user_text, final_response, intent, emo, rescored)
        # Log the updated profile after the turn
        updated_profile = profile_manager.load_profile(user_id)
        logger.debug(f"Updated Profile for {user_id}:\n{json.dumps(updated_profile, indent=2)}")

        return final_response
