from flask import Flask, request, jsonify
from src.core.chatbot import Chatbot
from src.core.profile_manager import ProfileManager
from src.config import config
import os

app = Flask(__name__)

# Initialize Chatbot and ProfileManager
chatbot = Chatbot()
profile_manager = ProfileManager(profile_dir=config.get("general.profile_dir"))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data.get('user_id')
    user_input = data.get('user_input')

    if not user_id or not user_input:
        return jsonify({'error': 'user_id and user_input are required'}), 400

    response = chatbot.handle_turn(user_id, user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)