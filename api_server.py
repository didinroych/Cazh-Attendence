from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from detect import FaceDetectionRecognition  # ⬅️ import class dari detect.py

app = Flask(__name__)

# Inisialisasi face detector
recognizer = FaceDetectionRecognition("embeddings.pkl")

# Route untuk halaman utama
@app.route("/")
def index():
    return render_template("live-cam.html")

@app.route("/user-list")
def user_list():
    return render_template("user-list.html")

@app.route("/regist-new-face")
def capture():
    return render_template("capture.html")

@app.route("/get-user-face", methods=["GET"])
def get_user_face():
    names = list(recognizer.embeddings_db.keys())
    return jsonify({"users": names})

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

@app.route("/register-face", methods=["POST"])
def register_face():
    file = request.files.get("video")
    name = request.form.get("name")

    if not file or not name:
        return jsonify({"error": "Missing name or video"}), 400

    # Save video file
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_name = name.replace(" ", "_")
    save_dir = os.path.join("video", safe_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestamp}.mp4")
    file.save(save_path)

    # Extract best face from video
    cap = cv2.VideoCapture(save_path)
    embeddings = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = recognizer.detect_faces(frame)
        if faces is not None and len(faces) > 0:
            face = faces[0]
            embedding = recognizer.get_face_embedding(frame, face)
            embeddings.append(embedding)
    cap.release()

    if len(embeddings) == 0:
        return jsonify({"error": "No face detected in video"}), 400

    mean_embedding = np.mean(embeddings, axis=0)
    recognizer.embeddings_db[name] = mean_embedding

    with open("embeddings.pkl", "wb") as f:
        pickle.dump(recognizer.embeddings_db, f)

    return jsonify({"status": "registered", "name": name, "frames_used": len(embeddings)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)