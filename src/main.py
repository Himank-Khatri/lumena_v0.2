import uuid
import uuid
import os # Import os module
import argparse # Import argparse for command-line arguments
from src.core.chatbot import Chatbot
from src.core.profile_manager import ProfileManager
from src.core.vector_store import vector_store
from src.utils import embed_semantic, embed_emotion
import json
from datetime import datetime
from src.logging import logger
from src.config import config

def initialize_kb_chunks():
    """Initializes some dummy KB chunks for demonstration."""
    dummy_chunks = [
        {
            "doc_id": str(uuid.uuid4()),
            "source": "CBT_Basics",
            "text": "Cognitive Behavioral Therapy (CBT) helps you identify and change negative thinking patterns.",
            "meta": {"therapy": "CBT", "topic": "introduction", "audience": "adult", "style": "informative"},
            "updated_at": datetime.now().isoformat()
        },
        {
            "doc_id": str(uuid.uuid4()),
            "source": "Mindfulness_Guide",
            "text": "Mindfulness involves focusing on the present moment without judgment. Try a 5-minute breathing exercise.",
            "meta": {"therapy": "Mindfulness", "topic": "exercise", "audience": "adult", "style": "gentle"},
            "updated_at": datetime.now().isoformat()
        },
        {
            "doc_id": str(uuid.uuid4()),
            "source": "Anxiety_Management",
            "text": "When feeling anxious, try deep breathing exercises. Inhale for 4 counts, hold for 4, exhale for 6.",
            "meta": {"therapy": "CBT", "topic": "anxiety", "audience": "adult", "style": "supportive"},
            "updated_at": datetime.now().isoformat()
        },
        {
            "doc_id": str(uuid.uuid4()),
            "source": "ACT_Principles",
            "text": "Acceptance and Commitment Therapy (ACT) encourages psychological flexibility and living by your values.",
            "meta": {"therapy": "ACT", "topic": "introduction", "audience": "adult", "style": "informative"},
            "updated_at": datetime.now().isoformat()
        },
        {
            "doc_id": str(uuid.uuid4()),
            "source": "Panic_Attack_Help",
            "text": "During a panic attack, focus on your senses: 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, 1 thing you can taste.",
            "meta": {"therapy": "CBT", "topic": "panic", "audience": "adult", "style": "direct"},
            "updated_at": datetime.now().isoformat()
        }
    ]

    # Add embeddings to dummy chunks before adding to vector store
    for chunk in dummy_chunks:
        chunk["embeddings"] = {
            "semantic": embed_semantic(chunk["text"]),
            "emotion": embed_emotion({"primary": "neutral", "scores": {"neutral": 1.0}}) # Default neutral for KB
        }
    vector_store.add_chunks(dummy_chunks)
    logger.debug(f"Initialized {len(dummy_chunks)} KB chunks.")

def main():
    logger.info("Initializing chatbot backend...")

    # Argument parsing
    parser = argparse.ArgumentParser(description="Lumena Chatbot")
    parser.add_argument("--user_id", type=str, default="default_user",
                        help="Specify a user ID to load or create a profile.")
    args = parser.parse_args()
    
    # Check if metadata.json exists
    if os.path.exists(vector_store.metadata_path):
        logger.info("metadata.json found. Clearing existing vector store before re-initialization.") # Reverted to print
        vector_store.clear_vector_store()
        logger.info("Existing vector store cleared based on metadata.json presence.")
    
    # Always initialize/re-initialize KB chunks
    initialize_kb_chunks()
    logger.info("KB chunks initialized/re-initialized.")
    
    chatbot = Chatbot()
    profile_manager = ProfileManager(profile_dir=config.get("general.profile_dir"))

    user_id = args.user_id
    
    # Create a default profile or load existing one
    profile = profile_manager.load_profile(user_id)
    logger.debug(f"Loaded Profile for {user_id}:\n{json.dumps(profile, indent=2)}")
    print(f"Loaded Profile for {user_id}.")

    print("\nStart chatting (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot.handle_turn(user_id, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
