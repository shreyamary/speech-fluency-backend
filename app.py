from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import speech_recognition as sr
import textstat
import openai
from pydub import AudioSegment
import os
import re
from langdetect import detect
from deep_translator import GoogleTranslator
import datetime

# Flask setup
app = Flask(__name__)
CORS(app)

# Database setup (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///speechfluency.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Filler words list
FILLERS = ["um", "uh", "like", "you know", "so", "actually", "basically"]

# ------------ Database Model -------------
class SpeechResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transcript = db.Column(db.Text, nullable=False)
    grammar_feedback = db.Column(db.Text, nullable=True)
    fluency_score = db.Column(db.Float, nullable=True)
    word_count = db.Column(db.Integer, nullable=True)
    wpm = db.Column(db.Float, nullable=True)
    fillers = db.Column(db.Text, nullable=True)  # comma-separated
    language = db.Column(db.String(10), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ------------ Helper Functions -------------
def get_gpt_feedback(text):
    prompt = f"Correct the grammar in this sentence and give friendly suggestions to improve spoken English:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def mentor_chat(message):
    prompt = f"You are a friendly spoken English coach. Reply to: {message}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def detect_fillers(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in FILLERS if w in words]

# ------------ API Routes -------------

@app.route("/analyze", methods=["POST"])
def analyze():
    audio_file = request.files.get('audio') or request.files.get('file')
    if not audio_file:
        return jsonify({"error": "No audio uploaded"}), 400

    # Convert to WAV
    audio = AudioSegment.from_file(audio_file)
    audio_path = "temp.wav"
    audio.export(audio_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            os.remove(audio_path)
            return jsonify({"error": "Speech not recognized"}), 400

    os.remove(audio_path)

    # Duration in seconds (passed from client)
    try:
        duration = float(request.form.get("duration", 0))
        if duration == 0:
            return jsonify({"error": "Invalid or zero duration"}), 400
    except ValueError:
        return jsonify({"error": "Invalid duration"}), 400

    duration_minutes = duration / 60
    word_count = len(transcript.split())
    wpm = round(word_count / duration_minutes, 2)
    score = textstat.flesch_reading_ease(transcript)
    grammar_feedback = get_gpt_feedback(transcript)
    detected_fillers = detect_fillers(transcript)
    lang = detect(transcript)

    # Save to database
    result = SpeechResult(
        transcript=transcript,
        grammar_feedback=grammar_feedback,
        fluency_score=score,
        word_count=word_count,
        wpm=wpm,
        fillers=",".join(detected_fillers),
        language=lang
    )
    db.session.add(result)
    db.session.commit()

    return jsonify({
        "transcript": transcript,
        "language": lang,
        "word_count": word_count,
        "wpm": wpm,
        "fluency_score": score,
        "fillers": detected_fillers,
        "grammar_feedback": grammar_feedback
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    reply = mentor_chat(message)
    return jsonify({"reply": reply})


# ------------ App Runner -------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
