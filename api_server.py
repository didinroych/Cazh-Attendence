from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from detect import FaceDetectionRecognition  # ⬅️ import class dari detect.py

app = Flask(__name__)

# Inisialisasi face detector
recognizer = FaceDetectionRecognition("embeddings.pkl")

# Route untuk halaman utama
@app.route("/")
def index():
    return render_template("live-cam.html")

@app.route("/verify-face", methods=["POST"])
def verify_face():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    processed_frame, faces_info = recognizer.process_frame(frame)
    
    if not faces_info:
        return jsonify({"status": "no face detected"}), 200
    
    return jsonify({
        "status": "ok",
        "name": faces_info[0]["name"],
        "confidence": faces_info[0]["score"],
        "spoof": faces_info[0]["score"] < recognizer.recognition_threshold
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)