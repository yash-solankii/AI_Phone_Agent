import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # fallback if dotenv not installed
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# API credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERVER_URL = os.getenv("SERVER_URL")
SERVER_PORT = int(os.getenv("PORT", "8080"))

# AI models
LLM_MODEL = "llama-3.1-8b-instant"
STT_MODEL = "whisper-large-v3"
TTS_MODEL = "playai-tts"
TTS_VOICE = "Jennifer-PlayAI"

# Audio settings
AUDIO_SAMPLE_RATE = 8000
VAD_FRAME_MS = 20
VAD_AGGRESSIVENESS = 1  # 1-3, higher = more aggressive
MIN_AUDIO_LEVEL_THRESHOLD = 0.015

# Timing thresholds
VAD_SILENCE_MS = 600  # silence before ending utterance
VAD_MIN_SPEECH_MS = 150  # min speech duration
ECHO_CANCELLATION_MS = 100  # ignore echo from agent
AGENT_RESPONSE_DELAY_MS = 100  # pause before responding
MAX_UTTERANCE_LENGTH_MS = 10000
MIN_MEANINGFUL_WORDS = 2

# Call limits
MAX_CALL_DURATION_S = 600  # 10 minutes
MAX_CONCURRENT_CALLS = 5
RATE_LIMIT_WINDOW_MINUTES = 1
RATE_LIMIT_CALLS_PER_WINDOW = 10