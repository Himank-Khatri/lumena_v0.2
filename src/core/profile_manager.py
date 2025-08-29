import json
import os
from datetime import datetime
import uuid
from src.config import config

class ProfileManager:
    def __init__(self, profile_dir=config.get("general.profile_dir")):
        self.profile_dir = profile_dir
        os.makedirs(self.profile_dir, exist_ok=True)

    def _get_profile_path(self, user_id: str):
        return os.path.join(self.profile_dir, f"{user_id}.json")

    def load_profile(self, user_id: str):
        """Loads a user profile from a JSON file."""
        profile_path = self._get_profile_path(user_id)
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                return json.load(f)
        else:
            # Return a default cold-start profile
            return self._create_default_profile(user_id)

    def save_profile(self, profile: dict):
        """Saves a user profile to a JSON file."""
        user_id = profile.get("user_id")
        if not user_id:
            raise ValueError("Profile must have a 'user_id'.")
        profile_path = self._get_profile_path(user_id)
        profile["updated_at"] = datetime.now().isoformat()
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)

    def upsert_profile(self, user_id: str, patch: dict):
        """Updates an existing profile or creates a new one with a patch."""
        current_profile = self.load_profile(user_id)
        # Merge the patch into the current profile
        # This is a shallow merge, for deeper merges, a more sophisticated logic is needed
        new_profile = {**current_profile, **patch}
        self.save_profile(new_profile)
        return new_profile

    def _create_default_profile(self, user_id: str):
        """Creates a default cold-start profile for a new user."""
        return {
            "user_id": user_id,
            "demographics": {},
            "conditions": [],
            "preferences": {"tone": "genuine", "length": "short", "avoid_topics": []},
            "goals": [],
            "triggers": [],
            "therapeutic_styles": ["CBT", "Mindfulness"],
            "timezone": "UTC", # Default, can be updated
            "safety_plan": {"contacts": [], "grounding_preference": ""},
            "consents": {"data_use": True, "crisis_detection": True},
            "updated_at": datetime.now().isoformat()
        }

    def profile_filters(self, profile: dict):
        """Extracts filters from the user profile for vector search."""
        filters = {
            "meta.therapy": profile.get("therapeutic_styles", ["CBT", "Mindfulness"]),
            "audience": "adult", # Assuming adult audience for now
            "language": "english" # Assuming English for now
        }
        return filters

    def profile_tags(self, profile: dict):
        """Extracts relevant tags from the user profile for scoring."""
        tags = []
        if profile.get("preferences", {}).get("tone"):
            tags.append(profile["preferences"]["tone"])
        tags.extend(profile.get("therapeutic_styles", []))
        tags.extend(profile.get("conditions", []))
        tags.extend(profile.get("goals", []))
        return tags
