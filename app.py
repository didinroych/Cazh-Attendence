from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import pickle
import json
from datetime import datetime,date, timedelta
from detect import FaceDetectionRecognition  

app = Flask(__name__)

recognizer = FaceDetectionRecognition("embeddings.pkl")
ATTENDANCE_FILE = "attendance.json"

def load_attendance_data():
    """Load attendance data from JSON file"""
    if not os.path.exists(ATTENDANCE_FILE):
        return {}
    
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_attendance_data(data):
    """Save attendance data to JSON file"""
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def get_today_string():
    """Get today's date as string"""
    return datetime.now().strftime("%Y-%m-%d")

@app.route("/")
def index():
    return render_template("live-cam.html")

@app.route("/user-list")
def user_list():
    return render_template("user-list.html")

@app.route("/regist-new-face")
def capture():
    return render_template("capture.html")

@app.route("/absensi")
def absent():
    return render_template("absensi.html")

@app.route("/get-user-face", methods=["GET"])
def get_user_face():
    users = []
    for uid, face_data in recognizer.embeddings_db.items():
        users.append({
            "uid_face": face_data["uid_face"],
            "name": face_data["name"]
        })
    return jsonify({"users": users})

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
    
    face_info = faces_info[0]  
    
    return jsonify({
        "status": "ok",
        "name": face_info["name"],
        "confidence": face_info["score"],
        "uid_face": face_info["uid_face"], 
        "spoof": face_info["score"] < recognizer.recognition_threshold
    })

@app.route("/register-face", methods=["POST"])
def register_face():
    file = request.files.get("video")
    name = request.form.get("name")
    uid_face = request.form.get("uid_face")  # Optional

    if not file or not name:
        return jsonify({"error": "Missing name or video"}), 400

    # Generate UID if not provided
    if not uid_face:
        uid_face = recognizer.generate_uid_face()
        while int(uid_face) in recognizer.embeddings_db:
            uid_face = recognizer.generate_uid_face()
    else:
        uid_face = int(uid_face)
        if uid_face in recognizer.embeddings_db:
            return jsonify({"error": "UID already exists"}), 400

    # Save video and get best frame
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_name = name.replace(" ", "_")
    save_dir = os.path.join("video", safe_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestamp}.mp4")
    file.save(save_path)

    # Get best face from video
    best_frame, best_face, quality_score = recognizer.get_best_face_from_video(save_path)
    
    if best_frame is None:
        return jsonify({"error": "No face detected in video"}), 400

    # Get embedding from best frame
    embedding = recognizer.get_face_embedding(best_frame, best_face)
    
    # Store in new format
    recognizer.embeddings_db[uid_face] = {
        "name": name,
        "embeddings": embedding,
        "uid_face": uid_face
    }

    with open("embeddings.pkl", "wb") as f:
        pickle.dump(recognizer.embeddings_db, f)

    return jsonify({
        "status": "registered", 
        "name": name, 
        "uid_face": uid_face,
        "quality_score": float(quality_score)
    })

@app.route("/register-face-average", methods=["POST"])
def register_face_average():
    file = request.files.get("video")
    name = request.form.get("name")
    uid_face = request.form.get("uid_face")  
    if not file or not name:
        return jsonify({"error": "Missing name or video"}), 400

    # Generate UID if not provided
    if not uid_face:
        uid_face = recognizer.generate_uid_face()
        # Ensure uniqueness
        while int(uid_face) in recognizer.embeddings_db:
            uid_face = recognizer.generate_uid_face()
    else:
        uid_face = int(uid_face)
        # Check if UID already exists
        if uid_face in recognizer.embeddings_db:
            return jsonify({"error": "UID already exists"}), 400

    # Save video file
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_name = name.replace(" ", "_")
    save_dir = os.path.join("video", safe_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestamp}.mp4")
    file.save(save_path)

    # Extract all face embeddings from video
    cap = cv2.VideoCapture(save_path)
    embeddings = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = recognizer.detect_faces(frame)
        if faces is not None and len(faces) > 0:
            face = faces[0]  # Take the first detected face
            embedding = recognizer.get_face_embedding(frame, face)
            embeddings.append(embedding)
        frame_count += 1
    cap.release()

    if len(embeddings) == 0:
        return jsonify({"error": "No face detected in video"}), 400

    # Calculate average embedding
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Store in new format
    recognizer.embeddings_db[uid_face] = {
        "name": name,
        "embeddings": mean_embedding,
        "uid_face": uid_face
    }

    # Save to file
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(recognizer.embeddings_db, f)

    return jsonify({
        "status": "registered", 
        "name": name, 
        "uid_face": uid_face,
        "frames_used": len(embeddings),
        "total_frames": frame_count
    })

@app.route("/clock-in", methods=["POST"])
def clock_in():
    """Clock in endpoint - requires face verification"""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Process face recognition
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    processed_frame, faces_info = recognizer.process_frame(frame)
    
    if not faces_info:
        return jsonify({"error": "No face detected"}), 400
    
    face_info = faces_info[0]
    
    # Check if face is recognized
    if face_info["name"] == "Unknown" or face_info["uid_face"] is None:
        return jsonify({"error": "Face not recognized"}), 400
    
    # Check for spoof
    if face_info["score"] < recognizer.recognition_threshold:
        return jsonify({"error": "Face verification failed - possible spoof"}), 400
    
    # Load attendance data
    attendance_data = load_attendance_data()
    today = get_today_string()
    uid_face = str(face_info["uid_face"])
    
    # Initialize user data if not exists
    if uid_face not in attendance_data:
        attendance_data[uid_face] = {}
    
    # Check if already clocked in today
    if today in attendance_data[uid_face]:
        if attendance_data[uid_face][today].get("clock_in_time"):
            return jsonify({
                "error": "Already clocked in today",
                "clock_in_time": attendance_data[uid_face][today]["clock_in_time"]
            }), 400
    
    # Record clock in
    current_time = datetime.now()
    
    attendance_data[uid_face][today] = {
        "name": face_info["name"],
        "clock_in_time": current_time.strftime("%H:%M:%S"),
        "clock_in_datetime": current_time.isoformat(),
        "clock_out_time": None,
        "clock_out_datetime": None,
        "duration": None,
        "status": "clocked_in"
    }
    
    # Save data
    save_attendance_data(attendance_data)
    
    return jsonify({
        "status": "success",
        "message": "Clock in successful",
        "name": face_info["name"],
        "uid_face": face_info["uid_face"],
        "clock_in_time": current_time.strftime("%H:%M:%S"),
        "date": today
    })

@app.route("/clock-out", methods=["POST"])
def clock_out():
    """Clock out endpoint - requires face verification"""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Process face recognition
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    processed_frame, faces_info = recognizer.process_frame(frame)
    
    if not faces_info:
        return jsonify({"error": "No face detected"}), 400
    
    face_info = faces_info[0]
    
    # Check if face is recognized
    if face_info["name"] == "Unknown" or face_info["uid_face"] is None:
        return jsonify({"error": "Face not recognized"}), 400
    
    # Check for spoof
    if face_info["score"] < recognizer.recognition_threshold:
        return jsonify({"error": "Face verification failed - possible spoof"}), 400
    
    # Load attendance data
    attendance_data = load_attendance_data()
    today = get_today_string()
    uid_face = str(face_info["uid_face"])
    
    # Check if user has clocked in today
    if uid_face not in attendance_data or today not in attendance_data[uid_face]:
        return jsonify({"error": "No clock in record found for today"}), 400
    
    today_record = attendance_data[uid_face][today]
    
    # Check if already clocked out
    if today_record.get("clock_out_time"):
        return jsonify({
            "error": "Already clocked out today",
            "clock_out_time": today_record["clock_out_time"],
            "duration": today_record["duration"]
        }), 400
    
    # Check if actually clocked in
    if not today_record.get("clock_in_time"):
        return jsonify({"error": "Must clock in first"}), 400
    
    # Record clock out
    current_time = datetime.now()
    clock_in_datetime = datetime.fromisoformat(today_record["clock_in_datetime"])
    
    # Calculate duration
    duration_seconds = (current_time - clock_in_datetime).total_seconds()
    duration_hours = duration_seconds / 3600
    duration_formatted = f"{int(duration_hours)}h {int((duration_seconds % 3600) / 60)}m"
    
    # Update record
    attendance_data[uid_face][today].update({
        "clock_out_time": current_time.strftime("%H:%M:%S"),
        "clock_out_datetime": current_time.isoformat(),
        "duration": duration_formatted,
        "duration_hours": round(duration_hours, 2),
        "status": "completed"
    })
    
    # Save data
    save_attendance_data(attendance_data)
    
    return jsonify({
        "status": "success",
        "message": "Clock out successful",
        "name": face_info["name"],
        "uid_face": face_info["uid_face"],
        "clock_in_time": today_record["clock_in_time"],
        "clock_out_time": current_time.strftime("%H:%M:%S"),
        "duration": duration_formatted,
        "duration_hours": round(duration_hours, 2),
        "date": today
    })

@app.route("/attendance-status", methods=["GET"])
def attendance_status():
    """Get attendance status for a user or all users"""
    uid_face = request.args.get("uid_face")
    date_param = request.args.get("date", get_today_string())
    
    attendance_data = load_attendance_data()
    
    if uid_face:
        # Get status for specific user
        uid_face = str(uid_face)
        
        if uid_face not in attendance_data or date_param not in attendance_data[uid_face]:
            return jsonify({
                "uid_face": uid_face,
                "date": date_param,
                "status": "not_clocked_in",
                "message": "No attendance record found"
            })
        
        record = attendance_data[uid_face][date_param]
        
        return jsonify({
            "uid_face": uid_face,
            "name": record["name"],
            "date": date_param,
            "status": record["status"],
            "clock_in_time": record.get("clock_in_time"),
            "clock_out_time": record.get("clock_out_time"),
            "duration": record.get("duration"),
            "duration_hours": record.get("duration_hours")
        })
    
    else:
        # Get status for all users for the specified date
        all_status = []
        
        for uid, user_data in attendance_data.items():
            if date_param in user_data:
                record = user_data[date_param]
                all_status.append({
                    "uid_face": uid,
                    "name": record["name"],
                    "date": date_param,
                    "status": record["status"],
                    "clock_in_time": record.get("clock_in_time"),
                    "clock_out_time": record.get("clock_out_time"),
                    "duration": record.get("duration"),
                    "duration_hours": record.get("duration_hours")
                })
        
        return jsonify({
            "date": date_param,
            "total_records": len(all_status),
            "attendance": all_status
        })
try:
    from api.swagger_config import init_swagger_api
    init_swagger_api(app, recognizer, load_attendance_data, save_attendance_data, get_today_string)
    print("✅ Swagger API documentation loaded at /api-docs/")
except ImportError:
    print("⚠️  Swagger API not available. Install flask-restx to enable API documentation.")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)