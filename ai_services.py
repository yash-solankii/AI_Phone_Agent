import io
import json
import logging
import wave  
from groq import Groq
from config import (LLM_MODEL, STT_MODEL, TTS_MODEL, TTS_VOICE,
                    AUDIO_SAMPLE_RATE, GROQ_API_KEY)
from session_manager import CallSession
from audio_utils import create_wav_bytes

groq_client = Groq(api_key=GROQ_API_KEY)

def synthesize_speech(text, interrupted_flag=None):
    """Convert text to speech audio"""
    try:
        # skip if already interrupted
        if interrupted_flag and interrupted_flag():
            return None
            
        tts_resp = groq_client.audio.speech.create(
            input=text, 
            voice=TTS_VOICE, 
            model=TTS_MODEL,
            response_format="wav", 
            sample_rate=AUDIO_SAMPLE_RATE
        )
        
        if interrupted_flag and interrupted_flag():
            return None
            
        # extract PCM data from WAV response
        with io.BytesIO(tts_resp.read()) as audio_io, wave.open(audio_io, 'rb') as wav_file:
            return wav_file.readframes(wav_file.getnframes())
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            logging.warning(f"TTS rate limit hit: {e}")
        else:
            logging.error(f"TTS failed: {e}", exc_info=True)
        return None

def transcribe_audio(audio_data):
    """Convert speech audio to text"""
    try:
        wav_bytes = create_wav_bytes(audio_data, AUDIO_SAMPLE_RATE)
        if not wav_bytes:
            return ""

        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"
        
        stt_prompt = (
            "Transcribe exactly what is spoken in this phone conversation. "
            "Be accurate and natural. If unclear or just noise, return empty string."
        )
        
        stt_resp = groq_client.audio.transcriptions.create(
            file=audio_file, 
            model=STT_MODEL,
            prompt=stt_prompt
        )
        user_text = stt_resp.text.strip()
        
        # filter out common Whisper hallucinations
        if user_text:
            hallucinations = [
                "thank you for calling",
                "how may i help you today",
                "is there anything else i can help you with",
                "have a great day and thank you for calling",
                "end of call",
                "call ended",
                "system message",
                "automated response"
            ]
            
            if any(phrase in user_text.lower() for phrase in hallucinations):
                logging.warning(f"Potential STT hallucination detected: '{user_text}'")
                return ""
            
            if len(user_text.strip()) < 3:
                logging.warning(f"Very short transcription, likely noise: '{user_text}'")
                return ""
        
        return user_text
    except Exception as e:
        logging.error(f"STT failed: {e}", exc_info=True)
        return ""

def generate_response(session, user_input):
    """Generate AI response to user input"""
    session.add_exchange(user_input, "")

    system_prompt = """You are Jennifer, a helpful AI assistant for phone conversations.
    
Be warm, natural, and conversational. Keep responses concise and human-like.
Use contractions and natural speech patterns.

Always respond in JSON format: {"action": "respond" or "hangup", "text": "your response"}

IMPORTANT: 
- Always provide a text response, never return null or empty text.
- If you don't know something, say so clearly instead of making things up.
- If the user asks about "our website" or "our company", ask them to clarify what they're referring to.
- Keep responses relevant to what the user actually asked.
- If ending the call, still provide a polite goodbye message in the text field.

Be helpful, ask for clarification when needed, and end calls naturally when appropriate."""

    messages = [
        {"role": "system", "content": system_prompt},
        *session.get_context(),
        {"role": "user", "content": user_input}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL, 
            messages=messages, 
            temperature=0.8,
            max_tokens=200, 
            response_format={"type": "json_object"}
        )
        response_text = response.choices[0].message.content
        response_json = json.loads(response_text)
        session.add_exchange(user_input, response_json.get("text", ""))
        return response_json
    except Exception as e:
        logging.error(f"LLM failed: {e}", exc_info=True)
        return {"action": "respond", "text": "Sorry, could you repeat that?"}