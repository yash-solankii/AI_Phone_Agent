import os
import signal
from flask import Flask
from flask_sock import Sock 
import config
from twilio_handler import register_routes

app = Flask(__name__)
sock = Sock(app)
register_routes(app, sock)

def shutdown_handler(signum, frame):
    print("\nShutting down...")
    os._exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == '__main__':
    print(f"Starting server on {config.SERVER_URL}")
    print(f"Webhook: {config.SERVER_URL}/voice")
    print(f"Phone: {config.TWILIO_PHONE_NUMBER}\n")
    app.run(host='0.0.0.0', port=config.SERVER_PORT, debug=False) 