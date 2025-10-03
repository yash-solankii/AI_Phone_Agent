import json
import logging
import queue
import threading
import time
import base64
import numpy as np
from twilio.twiml.voice_response import VoiceResponse, Connect
import webrtcvad

from config import (SERVER_URL, VAD_AGGRESSIVENESS, VAD_SILENCE_MS, VAD_MIN_SPEECH_MS, 
                    AUDIO_SAMPLE_RATE, VAD_FRAME_MS, ECHO_CANCELLATION_MS, 
                    MIN_AUDIO_LEVEL_THRESHOLD, MAX_UTTERANCE_LENGTH_MS, 
                    MIN_MEANINGFUL_WORDS, AGENT_RESPONSE_DELAY_MS)
from session_manager import CallSession
from audio_utils import decode_mulaw, pcm_to_ulaw
from ai_services import synthesize_speech, transcribe_audio, generate_response
from rate_limiter import rate_limiter
class AudioProcessor(threading.Thread):
    """Handles incoming audio, VAD, and outgoing audio streaming"""
    
    def __init__(self, session, ws):
        super().__init__()
        self.session = session
        self.ws = ws
        self.stop_event = threading.Event()
        self.incoming_audio_queue = queue.Queue()
        self.utterance_queue = queue.Queue()
        
        # voice activity detection
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.vad_frame_ms = VAD_FRAME_MS
        self.frame_bytes = int(AUDIO_SAMPLE_RATE * 2 * (self.vad_frame_ms / 1000))
        
        # state tracking
        self.last_agent_speech_time = 0
        self.utterance_start_time = None
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.is_sending_audio = False
        self.stop_audio_transmission = False

    def stop(self):
        self.stop_event.set()

    def add_incoming_audio(self, pcm_bytes):
        self.incoming_audio_queue.put(pcm_bytes)

    def send_audio_to_twilio(self, pcm_bytes):
        """Stream audio to Twilio in small chunks for fast interruption"""
        try:
            self.is_sending_audio = True
            self.stop_audio_transmission = False
            self.session.set_state("SPEAKING")
            self.last_agent_speech_time = time.time()
            
            # use 10ms chunks so interruption detection is super fast
            chunk_size = 160
            ulaw_data = pcm_to_ulaw(pcm_bytes)
            
            for i in range(0, len(ulaw_data), chunk_size):
                if self.stop_audio_transmission:
                    logging.info("Audio interrupted")
                    break
                    
                chunk = ulaw_data[i:i + chunk_size]
                ulaw_payload = base64.b64encode(chunk).decode('utf-8')
                message = {
                    "event": "media", 
                    "streamSid": self.session.call_sid, 
                    "media": {"payload": ulaw_payload}
                }
                self.ws.send(json.dumps(message))
                time.sleep(0.01)
            
            if not self.stop_audio_transmission:
                mark_message = {
                    "event": "mark", 
                    "streamSid": self.session.call_sid, 
                    "mark": {"name": "agent_speech_complete"}
                }
                self.ws.send(json.dumps(mark_message))
            self.is_sending_audio = False
        except Exception as e:
            logging.warning(f"Audio send failed: {e}")
            self.is_sending_audio = False

    def stop_speaking(self):
        """Immediately stop agent speech and clear Twilio's audio buffer"""
        try:
            self.stop_audio_transmission = True
            self.is_sending_audio = False
            
            # flush Twilio's buffer by sending multiple empty frames
            empty_audio = base64.b64encode(b'\xFF' * 160).decode('utf-8')
            for _ in range(5):
                stop_message = {
                    "event": "media", 
                    "streamSid": self.session.call_sid, 
                    "media": {"payload": empty_audio}
                }
                self.ws.send(json.dumps(stop_message))
            
            # tell Twilio to clear its queue
            clear_message = {"event": "clear", "streamSid": self.session.call_sid}
            self.ws.send(json.dumps(clear_message))
            
            mark_message = {
                "event": "mark", 
                "streamSid": self.session.call_sid, 
                "mark": {"name": "agent_speech_stopped"}
            }
            self.ws.send(json.dumps(mark_message))
            self.session.set_state("LISTENING")
            logging.info("Agent stopped")
        except Exception as e:
            logging.warning(f"Stop failed: {e}")

    def calculate_audio_level(self, frame):
        """Calculate RMS audio level (0.0 to 1.0)"""
        try:
            samples = np.frombuffer(frame, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            return rms / 32768.0
        except:
            return 0.0

    def is_meaningful_speech(self, frame):
        """Check if frame contains actual speech (not just noise)"""
        audio_level = self.calculate_audio_level(frame)
        if audio_level < MIN_AUDIO_LEVEL_THRESHOLD:
            return False
        return self.vad.is_speech(frame, AUDIO_SAMPLE_RATE)

    def run(self):
        """Main audio processing loop"""
        speech_buffer = bytearray()
        is_currently_speech = False
        silence_started_at = None
        min_speech_frames = VAD_MIN_SPEECH_MS // self.vad_frame_ms
        pause_tolerance_frames = 0
        max_pause_tolerance_frames = 10

        while not self.stop_event.is_set():
            try:
                frame = self.incoming_audio_queue.get(timeout=0.1)
                
                time_since_agent_speech = (time.time() - self.last_agent_speech_time) * 1000
                
                # check for user interruption - respond immediately
                is_speech = self.is_meaningful_speech(frame)
                if is_speech and self.session.agent_state == "SPEAKING":
                    self.consecutive_speech_frames += 1
                    if self.consecutive_speech_frames >= 1:  # instant detection
                        logging.info("User interrupted")
                        self.stop_speaking()
                        if hasattr(self, 'call_logic_ref'):
                            self.call_logic_ref.handle_interruption()
                        self.consecutive_speech_frames = 0
                else:
                    self.consecutive_speech_frames = 0
                
                # ignore echo from agent's own speech
                if time_since_agent_speech < ECHO_CANCELLATION_MS:
                    continue

                # build up utterances using VAD
                if is_speech:
                    self.consecutive_speech_frames += 1
                    self.consecutive_silence_frames = 0
                    pause_tolerance_frames = 0
                    
                    if not is_currently_speech:
                        self.utterance_start_time = time.time()
                    
                    speech_buffer.extend(frame)
                    is_currently_speech = True
                    silence_started_at = None
                else:
                    self.consecutive_silence_frames += 1
                    self.consecutive_speech_frames = 0
                    
                    if is_currently_speech:
                        # tolerate brief pauses within utterance
                        if pause_tolerance_frames < max_pause_tolerance_frames:
                            pause_tolerance_frames += 1
                            speech_buffer.extend(frame)
                            continue
                        
                        if silence_started_at is None:
                            silence_started_at = time.time()
                        
                        silence_duration = (time.time() - silence_started_at) * 1000
                        utterance_duration = (time.time() - self.utterance_start_time) * 1000 if self.utterance_start_time else 0
                        
                        # end utterance if silence is long enough or max length hit
                        if silence_duration > VAD_SILENCE_MS or utterance_duration > MAX_UTTERANCE_LENGTH_MS:
                            if len(speech_buffer) / self.frame_bytes > min_speech_frames:
                                self.utterance_queue.put(bytes(speech_buffer))
                            
                            speech_buffer.clear()
                            is_currently_speech = False
                            silence_started_at = None
                            self.utterance_start_time = None
                            pause_tolerance_frames = 0
                            
            except queue.Empty:
                if is_currently_speech:
                    if len(speech_buffer) / self.frame_bytes > min_speech_frames:
                        self.utterance_queue.put(bytes(speech_buffer))
                    
                    speech_buffer.clear()
                    is_currently_speech = False
                    silence_started_at = None
                    self.utterance_start_time = None
                    pause_tolerance_frames = 0
                continue

class CallLogic(threading.Thread):
    """Handles conversation flow - transcription, AI response, TTS"""
    
    def __init__(self, session, audio_processor):
        super().__init__()
        self.session = session
        self.audio_processor = audio_processor
        self.stop_event = threading.Event()
        self.pending_utterance = ""
        self.last_utterance_time = 0
        self.utterance_timeout = 3.0
        self.interrupted = False

    def stop(self):
        self.stop_event.set()

    def handle_interruption(self):
        """Called when user interrupts agent"""
        self.interrupted = True
        self.session.set_state("LISTENING")
        logging.info("Handling interruption")
        threading.Timer(0.1, lambda: setattr(self, 'interrupted', False)).start()

    def run(self):
        """Main conversation loop"""
        # say greeting
        greeting = "Hello, this is Jennifer. How can I help you today?"
        pcm_audio = synthesize_speech(greeting, lambda: self.interrupted)
        if pcm_audio and not self.interrupted:
            self.audio_processor.send_audio_to_twilio(pcm_audio)

        while not self.stop_event.is_set():
            try:
                utterance = self.audio_processor.utterance_queue.get(timeout=0.1)
                
                self.session.set_state("THINKING")
                user_text = transcribe_audio(utterance)
                
                # filter out noise and filler words
                if not user_text or len(user_text.strip()) < MIN_MEANINGFUL_WORDS:
                    self.session.set_state("LISTENING")
                    continue
                
                if user_text.lower().strip() in ['hmm', 'um', 'uh', 'ah', 'eh', 'oh']:
                    self.session.set_state("LISTENING")
                    continue
                
                # combine rapid-fire utterances
                current_time = time.time()
                if current_time - self.last_utterance_time < self.utterance_timeout:
                    user_text = f"{self.pending_utterance} {user_text}".strip()
                    self.pending_utterance = ""
                else:
                    # wait for complete sentences if user is still talking
                    if not user_text.endswith(('.', '!', '?')) and len(user_text.split()) < 5:
                        self.pending_utterance = user_text
                        self.last_utterance_time = current_time
                        self.session.set_state("LISTENING")
                        continue
                
                self.last_utterance_time = current_time
                logging.info(f"User: {user_text}")
                
                if self.interrupted:
                    self.interrupted = False
                    self.session.set_state("LISTENING")
                    continue
                
                # get AI response
                response_json = generate_response(self.session, user_text)
                
                if response_json:
                    action = response_json.get("action", "respond")
                    text_to_speak = response_json.get("text", "").strip()
                    
                    if text_to_speak and len(text_to_speak) > 2:
                        time.sleep(AGENT_RESPONSE_DELAY_MS / 1000.0)
                        
                        if self.interrupted:
                            self.interrupted = False
                            self.session.set_state("LISTENING")
                            continue
                        
                        # synthesize and play response
                        response_pcm = synthesize_speech(text_to_speak, lambda: self.interrupted)
                        if response_pcm and not self.interrupted:
                            self.audio_processor.send_audio_to_twilio(response_pcm)
                            logging.info(f"Agent: {text_to_speak}")
                        elif self.interrupted:
                            self.interrupted = False
                            self.session.set_state("LISTENING")
                        else:
                            logging.warning("TTS failed, ending call")
                            self.stop()
                    
                    if action == "hangup":
                        logging.info("Ending call")
                        threading.Timer(3.0, self.stop).start()

            except queue.Empty:
                # process pending utterance if timeout elapsed
                if self.pending_utterance and (time.time() - self.last_utterance_time) > self.utterance_timeout:
                    response_json = generate_response(self.session, self.pending_utterance)
                    if response_json and response_json.get("text"):
                        time.sleep(AGENT_RESPONSE_DELAY_MS / 1000.0)
                        response_pcm = synthesize_speech(response_json.get("text"))
                        if response_pcm:
                            self.audio_processor.send_audio_to_twilio(response_pcm)
                    self.pending_utterance = ""
                
                # check max call duration
                if self.session.should_end():
                    logging.info("Max call duration reached")
                    self.stop()
                continue
        
        logging.info("Call ended")
        self.audio_processor.stop()
        try:
            self.audio_processor.ws.close(1000, "Call ended")
        except:
            pass

def register_routes(app, sock):
    @app.route('/voice', methods=['POST'])
    def voice_webhook():
        """Handle incoming Twilio calls"""
        from flask import request
        caller_id = request.form.get('From', 'unknown')
        
        # check rate limits
        if not rate_limiter.can_start_call(caller_id):
            logging.warning(f"Rate limit hit: {caller_id}")
            response = VoiceResponse()
            response.say("Sorry, we're experiencing high call volume. Please try again later.")
            response.hangup()
            return str(response), 200, {'Content-Type': 'text/xml'}
        
        logging.info("Incoming call")
        response = VoiceResponse()
        connect = Connect()
        websocket_url = f"wss://{SERVER_URL.split('://')[1]}/ws"
        connect.stream(url=websocket_url)
        response.append(connect)
        return str(response), 200, {'Content-Type': 'text/xml'}

    @sock.route('/ws')
    def websocket_handler(ws):
        """Handle WebSocket connection for audio streaming"""
        logging.info("WebSocket connected")
        session, audio_processor, call_logic = None, None, None
        frame_buffer = bytearray()
        frame_bytes = int(AUDIO_SAMPLE_RATE * 2 * (20 / 1000))
        
        try:
            while True:
                message = ws.receive(timeout=1)
                if message is None:
                    continue
                    
                data = json.loads(message)
                
                if data['event'] == 'start':
                    # start new call
                    stream_sid = data['start']['streamSid']
                    session = CallSession(call_sid=stream_sid)
                    audio_processor = AudioProcessor(session, ws)
                    call_logic = CallLogic(session, audio_processor)
                    audio_processor.call_logic_ref = call_logic
                    audio_processor.start()
                    call_logic.start()
                    
                elif data['event'] == 'media':
                    # incoming audio from user
                    if audio_processor:
                        pcm_bytes = decode_mulaw(data['media']['payload'])
                        frame_buffer.extend(pcm_bytes)
                        # process in 20ms frames
                        while len(frame_buffer) >= frame_bytes:
                            frame = frame_buffer[:frame_bytes]
                            frame_buffer = frame_buffer[frame_bytes:]
                            audio_processor.add_incoming_audio(frame)
                            
                elif data['event'] == 'mark':
                    # audio playback markers
                    if session:
                        if data['mark']['name'] in ["agent_speech_complete", "agent_speech_stopped"]:
                            session.set_state("LISTENING")
                            
                elif data['event'] == 'stop':
                    logging.info("Call ended")
                    break
                    
        except Exception as e: 
            logging.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            if audio_processor: 
                audio_processor.stop()
            if call_logic: 
                call_logic.stop()
            rate_limiter.end_call()
            logging.info("WebSocket closed")