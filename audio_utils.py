import base64
import io
import wave
import logging
import numpy as np

# Convert PCM audio to mu-law format (Twilio expects mu-law)
def pcm_to_ulaw(pcm_data):
    samples = np.frombuffer(pcm_data, dtype=np.int16).copy()
    np.clip(samples, -32767, 32767, out=samples)
    
    ulaw = bytearray(len(samples))
    for i, sample in enumerate(samples):
        sign = 0x80 if sample < 0 else 0
        sample_abs = abs(sample)
        
        if sample_abs > 32635:
            sample_abs = 32635
        
        # find exponent
        exponent = 7
        mask = 0x4000
        while (sample_abs & mask) == 0 and exponent > 0:
            exponent -= 1
            mask >>= 1
        
        mantissa = (sample_abs >> (exponent + 3)) & 0x0F
        ulaw_byte = (sign | (exponent << 4) | mantissa) ^ 0xFF
        ulaw[i] = ulaw_byte
    
    return bytes(ulaw)

# Wrap raw PCM data in WAV format for Whisper
def create_wav_bytes(audio_data, sample_rate):
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        return wav_buffer.getvalue()
    except Exception as e:
        logging.error(f"WAV creation failed: {e}")
        return None

# Decode mu-law audio from Twilio to PCM
def decode_mulaw(payload):
    # mu-law to linear PCM lookup table
    _ulaw2lin = np.array([-32124,-31100,-30076,-29052,-28028,-27004,-25980,-24956,-23932,-22908,-21884,-20860,-19836,-18812,-17788,-16764,-15996,-15484,-14972,-14460,-13948,-13436,-12924,-12412,-11900,-11388,-10876,-10364,-9852,-9340,-8828,-8316,-7932,-7676,-7420,-7164,-6908,-6652,-6396,-6140,-5884,-5628,-5372,-5116,-4860,-4604,-4348,-4092,-3900,-3772,-3644,-3516,-3388,-3260,-3132,-3004,-2876,-2748,-2620,-2492,-2364,-2236,-2108,-1980,-1884,-1820,-1756,-1692,-1628,-1564,-1500,-1436,-1372,-1308,-1244,-1180,-1116,-1052,-988,-924,-876,-844,-812,-780,-748,-716,-684,-652,-620,-588,-556,-524,-492,-460,-428,-396,-372,-356,-340,-324,-308,-292,-276,-260,-244,-228,-212,-196,-180,-164,-148,-132,-128,-120,-112,-104,-96,-88,-80,-72,-64,-56,-48,-40,-32,-24,-16,-8,0,32124,31100,30076,29052,28028,27004,25980,24956,23932,22908,21884,20860,19836,18812,17788,16764,15996,15484,14972,14460,13948,13436,12924,12412,11900,11388,10876,10364,9852,9340,8828,8316,7932,7676,7420,7164,6908,6652,6396,6140,5884,5628,5372,5116,4860,4604,4348,4092,3900,3772,3644,3516,3388,3260,3132,3004,2876,2748,2620,2492,2364,2236,2108,1980,1884,1820,1756,1692,1628,1564,1500,1436,1372,1308,1244,1180,1116,1052,988,924,876,844,812,780,748,716,684,652,620,588,556,524,492,460,428,396,372,356,340,324,308,292,276,260,244,228,212,196,-180,-164,-148,-132,-128,-120,-112,-104,-96,-88,-80,-72,-64,-56,-48,-40,-32,-24,-16,-8,0], dtype=np.int16)
    ulaw_data = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    return _ulaw2lin[ulaw_data].tobytes()