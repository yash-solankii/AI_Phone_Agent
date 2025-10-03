# Twilio AI Phone Agent

A real-time AI phone agent that handles voice calls with natural conversation using Twilio, Groq AI, and WebSockets.

## Features

- **Voice Calls**: Handle incoming phone calls with real-time audio processing
- **AI Responses**: Powered by Groq's Llama 3 model for natural conversation
- **Speech Processing**: Voice Activity Detection (VAD) and echo cancellation
- **Fast Interruption**: User can interrupt the agent mid-speech (< 100ms response time)
- **Rate Limiting**: Built-in protection against spam and overload
- **WebSocket**: Real-time bidirectional audio streaming

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file:
```env
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
GROQ_API_KEY=your_groq_api_key
SERVER_URL=https://your-ngrok-url.ngrok-free.app
PORT=8080
```

### 3. Get API Keys

**Twilio** (Free trial available):
- Sign up at [twilio.com](https://www.twilio.com)
- Get Account SID, Auth Token, and phone number

**Groq** (Free tier available):
- Sign up at [groq.com](https://console.groq.com)
- Get API key from dashboard

**ngrok** (Free tier available):
- Download from [ngrok.com](https://ngrok.com)
- Run: `ngrok http 8080`

### 4. Run the Application
```bash
python app.py
```

## Usage

1. Start ngrok: `ngrok http 8080`
2. Update Twilio webhook URL to: `https://your-ngrok-url.ngrok-free.app/voice`
3. Call your Twilio phone number
4. Talk to Jennifer, the AI assistant

## Configuration

Key settings in `config.py`:
- `MAX_CONCURRENT_CALLS = 5` - Max simultaneous calls
- `RATE_LIMIT_CALLS_PER_WINDOW = 10` - Calls per minute per caller
- `VAD_SILENCE_MS = 600` - Silence detection threshold (ms)
- `ECHO_CANCELLATION_MS = 100` - Echo cancellation buffer (ms)
- `AGENT_RESPONSE_DELAY_MS = 100` - Pause before agent responds

## Architecture

```
Phone Call → Twilio → ngrok → Flask App → WebSocket → AI Processing
```

- **Flask App**: Main server with voice webhook
- **WebSocket**: Real-time audio streaming
- **Audio Processor**: VAD, echo cancellation, speech detection
- **AI Services**: Groq integration for STT, LLM, TTS
- **Rate Limiter**: Call protection and limits

## Files

- `app.py` - Main Flask application
- `twilio_handler.py` - Call handling and WebSocket logic
- `ai_services.py` - Groq AI integration
- `session_manager.py` - Call session management
- `audio_utils.py` - Audio format conversion
- `rate_limiter.py` - Rate limiting protection
- `config.py` - Configuration settings

## Troubleshooting

**No audio**: Check ngrok URL and Twilio webhook configuration  
**Slow responses**: Check Groq API limits and network connection  
**Rate limited**: Adjust `RATE_LIMIT_CALLS_PER_WINDOW` in config.py  
**Interruptions not working**: Check audio levels and `MIN_AUDIO_LEVEL_THRESHOLD` setting

## License

MIT License