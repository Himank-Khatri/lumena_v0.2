import os
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.config import config

load_dotenv()

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize models
# Emotion classification model
emotion_classifier = pipeline(
    "sentiment-analysis",
    model=config.get("embeddings.emotion_classifier_model"),
    top_k=None
)

# Semantic embedding model
semantic_embedder = SentenceTransformer(config.get("embeddings.semantic_model"))

def classify_emotion(text: str):
    """Classifies the emotion of the given text."""
    results = emotion_classifier(text)[0]
    # The pipeline returns a list of dicts, we want the top emotion
    primary_emotion = results[0]['label']
    scores = {res['label']: res['score'] for res in results}
    return {"primary": primary_emotion, "scores": scores}

def embed_semantic(text: str):
    """Generates semantic embeddings for the given text."""
    return semantic_embedder.encode(text).tolist()

def embed_emotion(emotion_data: dict):
    """Generates emotion embeddings based on the primary emotion."""
    # For simplicity, we'll use a one-hot encoding or a simple vector based on primary emotion
    # In a real scenario, this might be a more sophisticated model or a learned embedding.
    # For now, let's create a dummy embedding based on the primary emotion.
    # This needs to be consistent in size with semantic embeddings for scoring.
    # Let's assume a fixed size, e.g., 384 (same as all-MiniLM-L6-v2)
    embedding_size = semantic_embedder.get_sentence_embedding_dimension()
    emotion_vector = np.zeros(embedding_size)
    # A very basic approach: assign a value to a specific index based on emotion
    # This is a placeholder and should be replaced with a more robust method.
    emotion_map = {
        "anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4,
        "sadness": 5, "surprise": 6, "anxious": 7, "hopeful": 8
    }
    if emotion_data["primary"] in emotion_map:
        emotion_vector[emotion_map[emotion_data["primary"]] % embedding_size] = 1.0
    return emotion_vector.tolist()

def cosine_similarity_score(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]

# Placeholder for intent classification (can be expanded with a custom model or LLM call)
def classify_intent(user_text: str):
    """Classifies the intent of the user's message."""
    # This is a simplified placeholder. In a real system, this would be an ML model or LLM call.
    user_text_lower = user_text.lower()
    if "feel" in user_text_lower or "feeling" in user_text_lower:
        return "share_feelings"
    elif "what is" in user_text_lower or "tell me about" in user_text_lower:
        return "ask_info"
    elif "exercise" in user_text_lower or "practice" in user_text_lower:
        return "exercise"
    else:
        return "general_chat"

# Placeholder for safety screening
def screen_safety(user_text: str, emo: dict):
    """Screens user text for safety concerns."""
    # This is a simplified placeholder. In a real system, this would be a robust ML model.
    safety_level = "none"
    if "harm myself" in user_text.lower() or "suicide" in user_text.lower():
        safety_level = "crisis"
    return {"level": safety_level, "self_harm": 0.0} # Placeholder score
