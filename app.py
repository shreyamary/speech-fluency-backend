from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
import textstat
import openai
from pydub import AudioSegment
import os
from langdetect import detect
from googletrans import Translator

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in Render later

translator = Translator()

# Filler words list
FILLERS = ["um", "uh", "like", "you know", "so", "actually", "basically"]

# 🔹 Helper: GPT grammar feedback
def get_gpt_feedback(text):
    prompt = f"Correct the grammar in this sentence and give friendly suggestions to improve spoken English:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 🔹 Helper: AI Mentor Chatbot
def mentor_chat(message):
    prompt = f"You are a friendly spoken English coach. Reply to: {message}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 🔹 Endpoint: Analyze Speech
@app.route("/analyze", methods=["POST"])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files['audio']
    audio_path = "temp.wav"

    # Convert to WAV if needed
    audio = AudioSegment.from_file(audio_file)
    audio.export(audio_path, format="wav")

    # Use SpeechRecognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return jsonify({"error": "Speech not recognized"}), 400

    # Clean up temp file
    os.remove(audio_path)

    # Calculate features
    word_count = len(transcript.split())
    duration_minutes = float(request.form.get("duration", 1)) / 60
    wpm = round(word_count / duration_minutes, 2)
    score = textstat.flesch_reading_ease(transcript)
    grammar_feedback = get_gpt_feedback(transcript)
    detected_fillers = [w for w in FILLERS if w in transcript.lower()]
    lang = detect(transcript)

    return jsonify({
        "transcript": transcript,
        "language": lang,
        "word_count": word_count,
        "wpm": wpm,
        "fluency_score": score,
        "fillers": detected_fillers,
        "grammar_feedback": grammar_feedback
    })

# 🔹 Endpoint: AI Mentor Chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    reply = mentor_chat(message)
    return jsonify({"reply": reply})

# 🔹 Endpoint: Translation
@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    target_lang = data.get("to", "en")
    try:
        translated = translator.translate(text, dest=target_lang)
        return jsonify({"translated": translated.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Home route
@app.route("/", methods=["GET"])
def home():
    return "Speech Fluency Coach API is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
