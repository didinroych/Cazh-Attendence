"""
Swagger API Configuration for Face Recognition Attendance System
Fixed version with proper error handling
"""

from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
import cv2
import numpy as np
import os
import pickle
from datetime import datetime

def init_swagger_api(app, recognizer, load_attendance_data, save_attendance_data, get_today_string):
    """Initialize Swagger API with Flask-RESTX"""
    
    # Initialize Flask-RESTX for Swagger
    api = Api(
        app,
        version='1.0',
        title='Face Recognition Attendance API',
        description='A comprehensive API for face recognition and attendance management system',
        doc='/api-docs/',
        prefix='/api'
    )
    
    # Create namespaces for better organization
    ns_face = api.namespace('face', description='Face Recognition Operations')
    ns_attendance = api.namespace('attendance', description='Attendance Management')
    ns_users = api.namespace('users', description='User Management')
    
    # Define API models for Swagger documentation
    face_verification_model = api.model('FaceVerificationResponse', {
        'status': fields.String(required=True, description='Verification status', example='ok'),
        'name': fields.String(required=True, description='Recognized person name', example='John Doe'),
        'confidence': fields.Float(required=True, description='Recognition confidence score', example=0.95),
        'uid_face': fields.Integer(required=True, description='Unique face ID', example=12345678),
        'spoof': fields.Boolean(required=True, description='Whether spoof is detected', example=False)
    })
    
    user_model = api.model('User', {
        'uid_face': fields.Integer(required=True, description='Unique face ID', example=12345678),
        'name': fields.String(required=True, description='User name', example='John Doe')
    })
    
    users_response_model = api.model('UsersResponse', {
        'users': fields.List(fields.Nested(user_model), required=True, description='List of registered users')
    })
    
    face_registration_response = api.model('FaceRegistrationResponse', {
        'status': fields.String(required=True, description='Registration status', example='registered'),
        'name': fields.String(required=True, description='Registered person name', example='John Doe'),
        'uid_face': fields.Integer(required=True, description='Generated unique face ID', example=12345678),
        'quality_score': fields.Float(required=True, description='Face quality score', example=1250.5)
    })
    
    face_registration_average_response = api.model('FaceRegistrationAverageResponse', {
        'status': fields.String(required=True, description='Registration status', example='registered'),
        'name': fields.String(required=True, description='Registered person name', example='John Doe'),
        'uid_face': fields.Integer(required=True, description='Generated unique face ID', example=12345678),
        'frames_used': fields.Integer(required=True, description='Number of frames used for averaging', example=45),
        'total_frames': fields.Integer(required=True, description='Total frames in video', example=150)
    })
    
    attendance_record_model = api.model('AttendanceRecord', {
        'uid_face': fields.String(required=True, description='User face ID', example='12345678'),
        'name': fields.String(required=True, description='User name', example='John Doe'),
        'date': fields.String(required=True, description='Date of attendance', example='2025-06-30'),
        'status': fields.String(required=True, description='Attendance status', example='completed'),
        'clock_in_time': fields.String(description='Clock in time', example='08:30:00'),
        'clock_out_time': fields.String(description='Clock out time', example='17:45:00'),
        'duration': fields.String(description='Work duration', example='9h 15m'),
        'duration_hours': fields.Float(description='Duration in hours', example=9.25)
    })
    
    attendance_status_response = api.model('AttendanceStatusResponse', {
        'date': fields.String(required=True, description='Query date', example='2025-06-30'),
        'total_records': fields.Integer(required=True, description='Total attendance records', example=5),
        'attendance': fields.List(fields.Nested(attendance_record_model), required=True, description='Attendance records')
    })
    
    clock_response = api.model('ClockResponse', {
        'status': fields.String(required=True, description='Operation status', example='success'),
        'message': fields.String(required=True, description='Success message', example='Clock in successful'),
        'name': fields.String(required=True, description='User name', example='John Doe'),
        'uid_face': fields.Integer(required=True, description='User face ID', example=12345678),
        'clock_in_time': fields.String(description='Clock in time', example='08:30:00'),
        'clock_out_time': fields.String(description='Clock out time', example='17:45:00'),
        'duration': fields.String(description='Work duration', example='9h 15m'),
        'duration_hours': fields.Float(description='Duration in hours', example=9.25),
        'date': fields.String(required=True, description='Date', example='2025-06-30')
    })
    
    error_model = api.model('Error', {
        'error': fields.String(required=True, description='Error message', example='No image uploaded')
    })
    
    # File upload parsers
    upload_parser = reqparse.RequestParser()
    upload_parser.add_argument('image', location='files', type=FileStorage, required=True, help='Image file for face verification')
    
    video_upload_parser = reqparse.RequestParser()
    video_upload_parser.add_argument('video', location='files', type=FileStorage, required=True, help='Video file for face registration')
    video_upload_parser.add_argument('name', type=str, required=True, help='Person name for registration')
    video_upload_parser.add_argument('uid_face', type=int, required=False, help='Optional custom face ID')
    
    attendance_query_parser = reqparse.RequestParser()
    attendance_query_parser.add_argument('uid_face', type=str, required=False, help='Specific user face ID')
    attendance_query_parser.add_argument('date', type=str, required=False, help='Date in YYYY-MM-DD format (default: today)')
    
    # Face Recognition API Endpoints
    @ns_face.route('/verify')
    class FaceVerification(Resource):
        @ns_face.doc('verify_face')
        @ns_face.expect(upload_parser)
        @ns_face.marshal_with(face_verification_model)
        @ns_face.response(400, 'Bad Request', error_model)
        def post(self):
            """
            Verify a face from uploaded image
            
            Upload an image file to verify if the face is recognized.
            Returns person information if face is recognized.
            """
            try:
                args = upload_parser.parse_args()
                file = args['image']
                
                if not file:
                    return {"error": "No image uploaded"}, 400
                
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return {"error": "Invalid image format"}, 400
                
                processed_frame, faces_info = recognizer.process_frame(frame)
                
                if not faces_info:
                    return {"status": "no face detected", "name": "", "confidence": 0.0, "uid_face": 0, "spoof": False}
                
                face_info = faces_info[0]
                
                return {
                    "status": "ok",
                    "name": face_info["name"],
                    "confidence": face_info["score"],
                    "uid_face": face_info["uid_face"] or 0,
                    "spoof": face_info["score"] < recognizer.recognition_threshold
                }
            except Exception as e:
                return {"error": f"Processing error: {str(e)}"}, 500
    
    @ns_face.route('/register')
    class FaceRegistration(Resource):
        @ns_face.doc('register_face')
        @ns_face.expect(video_upload_parser)
        @ns_face.marshal_with(face_registration_response)
        @ns_face.response(400, 'Bad Request', error_model)
        def post(self):
            """
            Register a new face using video (best frame method)
            
            Upload a video file and person name to register a new face.
            Uses the best quality frame from the video for registration.
            """
            try:
                args = video_upload_parser.parse_args()
                file = args['video']
                name = args['name']
                uid_face = args.get('uid_face')

                if not file or not name:
                    return {"error": "Missing name or video"}, 400

                # Generate UID if not provided
                if not uid_face:
                    uid_face = recognizer.generate_uid_face()
                    while int(uid_face) in recognizer.embeddings_db:
                        uid_face = recognizer.generate_uid_face()
                else:
                    if uid_face in recognizer.embeddings_db:
                        return {"error": "UID already exists"}, 400

                # Save video and process
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                safe_name = name.replace(" ", "_")
                save_dir = os.path.join("video", safe_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{timestamp}.mp4")
                file.save(save_path)

                # Get best face from video
                best_frame, best_face, quality_score = recognizer.get_best_face_from_video(save_path)
                
                if best_frame is None:
                    return {"error": "No face detected in video"}, 400

                # Get embedding and save
                embedding = recognizer.get_face_embedding(best_frame, best_face)
                
                recognizer.embeddings_db[uid_face] = {
                    "name": name,
                    "embeddings": embedding,
                    "uid_face": uid_face
                }

                with open("embeddings.pkl", "wb") as f:
                    pickle.dump(recognizer.embeddings_db, f)

                return {
                    "status": "registered", 
                    "name": name, 
                    "uid_face": uid_face,
                    "quality_score": float(quality_score)
                }
            except Exception as e:
                return {"error": f"Registration error: {str(e)}"}, 500

    @ns_face.route('/register-average')
    class FaceRegistrationAverage(Resource):
        @ns_face.doc('register_face_average')
        @ns_face.expect(video_upload_parser)
        @ns_face.marshal_with(face_registration_average_response)
        @ns_face.response(400, 'Bad Request', error_model)
        def post(self):
            """
            Register a new face using video (average embedding method)
            
            Upload a video file and person name to register a new face.
            Uses averaged embeddings from all detected faces in the video for better accuracy.
            """
            try:
                args = video_upload_parser.parse_args()
                file = args['video']
                name = args['name']
                uid_face = args.get('uid_face')
                
                if not file or not name:
                    return {"error": "Missing name or video"}, 400

                # Generate UID if not provided
                if not uid_face:
                    uid_face = recognizer.generate_uid_face()
                    while int(uid_face) in recognizer.embeddings_db:
                        uid_face = recognizer.generate_uid_face()
                else:
                    if uid_face in recognizer.embeddings_db:
                        return {"error": "UID already exists"}, 400

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
                        face = faces[0]
                        embedding = recognizer.get_face_embedding(frame, face)
                        embeddings.append(embedding)
                    frame_count += 1
                cap.release()

                if len(embeddings) == 0:
                    return {"error": "No face detected in video"}, 400

                # Calculate average embedding
                mean_embedding = np.mean(embeddings, axis=0)
                
                # Store in database
                recognizer.embeddings_db[uid_face] = {
                    "name": name,
                    "embeddings": mean_embedding,
                    "uid_face": uid_face
                }

                with open("embeddings.pkl", "wb") as f:
                    pickle.dump(recognizer.embeddings_db, f)

                return {
                    "status": "registered", 
                    "name": name, 
                    "uid_face": uid_face,
                    "frames_used": len(embeddings),
                    "total_frames": frame_count
                }
            except Exception as e:
                return {"error": f"Registration error: {str(e)}"}, 500

    # User Management API Endpoints
    @ns_users.route('/list')
    class UserList(Resource):
        @ns_users.doc('get_users')
        @ns_users.marshal_with(users_response_model)
        def get(self):
            """
            Get list of all registered users
            
            Returns a list of all users registered in the face recognition system
            with their names and unique face IDs.
            """
            try:
                users = []
                for uid, face_data in recognizer.embeddings_db.items():
                    users.append({
                        "uid_face": face_data["uid_face"],
                        "name": face_data["name"]
                    })
                return {"users": users}
            except Exception as e:
                return {"error": f"Error retrieving users: {str(e)}"}, 500

    # Attendance Management API Endpoints
    @ns_attendance.route('/clock-in')
    class ClockIn(Resource):
        @ns_attendance.doc('clock_in')
        @ns_attendance.expect(upload_parser)
        @ns_attendance.marshal_with(clock_response)
        @ns_attendance.response(400, 'Bad Request', error_model)
        def post(self):
            """
            Clock in with face verification
            
            Upload an image for face verification to clock in.
            Face must be recognized and not spoofed. User cannot clock in twice on the same day.
            """
            try:
                args = upload_parser.parse_args()
                file = args['image']
                
                if not file:
                    return {"error": "No image uploaded"}, 400
                
                # Process face recognition
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return {"error": "Invalid image format"}, 400
                
                processed_frame, faces_info = recognizer.process_frame(frame)
                
                if not faces_info:
                    return {"error": "No face detected"}, 400
                
                face_info = faces_info[0]
                
                # Check if face is recognized
                if face_info["name"] == "Unknown" or face_info["uid_face"] is None:
                    return {"error": "Face not recognized"}, 400
                
                # Check for spoof
                if face_info["score"] < recognizer.recognition_threshold:
                    return {"error": "Face verification failed - possible spoof"}, 400
                
                # Process attendance
                attendance_data = load_attendance_data()
                today = get_today_string()
                uid_face = str(face_info["uid_face"])
                
                # Initialize user data if not exists
                if uid_face not in attendance_data:
                    attendance_data[uid_face] = {}
                
                # Check if already clocked in today
                if today in attendance_data[uid_face]:
                    if attendance_data[uid_face][today].get("clock_in_time"):
                        return {
                            "error": f"Already clocked in today at {attendance_data[uid_face][today]['clock_in_time']}"
                        }, 400
                
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
                
                save_attendance_data(attendance_data)
                
                return {
                    "status": "success",
                    "message": "Clock in successful",
                    "name": face_info["name"],
                    "uid_face": face_info["uid_face"],
                    "clock_in_time": current_time.strftime("%H:%M:%S"),
                    "date": today
                }
            except Exception as e:
                return {"error": f"Clock in error: {str(e)}"}, 500

    @ns_attendance.route('/clock-out')
    class ClockOut(Resource):
        @ns_attendance.doc('clock_out')
        @ns_attendance.expect(upload_parser)
        @ns_attendance.marshal_with(clock_response)
        @ns_attendance.response(400, 'Bad Request', error_model)
        def post(self):
            """
            Clock out with face verification
            
            Upload an image for face verification to clock out.
            User must have clocked in first and cannot clock out twice on the same day.
            """
            try:
                args = upload_parser.parse_args()
                file = args['image']
                
                if not file:
                    return {"error": "No image uploaded"}, 400
                
                # Process face recognition
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return {"error": "Invalid image format"}, 400
                
                processed_frame, faces_info = recognizer.process_frame(frame)
                
                if not faces_info:
                    return {"error": "No face detected"}, 400
                
                face_info = faces_info[0]
                
                # Check if face is recognized
                if face_info["name"] == "Unknown" or face_info["uid_face"] is None:
                    return {"error": "Face not recognized"}, 400
                
                # Check for spoof
                if face_info["score"] < recognizer.recognition_threshold:
                    return {"error": "Face verification failed - possible spoof"}, 400
                
                # Process attendance
                attendance_data = load_attendance_data()
                today = get_today_string()
                uid_face = str(face_info["uid_face"])
                
                # Check if user has clocked in today
                if uid_face not in attendance_data or today not in attendance_data[uid_face]:
                    return {"error": "No clock in record found for today"}, 400
                
                today_record = attendance_data[uid_face][today]
                
                # Check if already clocked out
                if today_record.get("clock_out_time"):
                    return {
                        "error": f"Already clocked out today at {today_record['clock_out_time']}"
                    }, 400
                
                # Check if actually clocked in
                if not today_record.get("clock_in_time"):
                    return {"error": "Must clock in first"}, 400
                
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
                
                save_attendance_data(attendance_data)
                
                return {
                    "status": "success",
                    "message": "Clock out successful",
                    "name": face_info["name"],
                    "uid_face": face_info["uid_face"],
                    "clock_in_time": today_record["clock_in_time"],
                    "clock_out_time": current_time.strftime("%H:%M:%S"),
                    "duration": duration_formatted,
                    "duration_hours": round(duration_hours, 2),
                    "date": today
                }
            except Exception as e:
                return {"error": f"Clock out error: {str(e)}"}, 500

    @ns_attendance.route('/status')
    class AttendanceStatus(Resource):
        @ns_attendance.doc('attendance_status')
        @ns_attendance.expect(attendance_query_parser)
        @ns_attendance.marshal_with(attendance_status_response)
        def get(self):
            """
            Get attendance status
            
            Get attendance status for a specific user or all users for a given date.
            If no parameters provided, returns all users' status for today.
            """
            try:
                args = attendance_query_parser.parse_args()
                uid_face = args.get('uid_face')
                date_param = args.get('date') or get_today_string()  # ✅ Fix: Use 'or' instead of get() default
                
                print(f"Debug - uid_face: {uid_face}, date_param: {date_param}")  # Debug line
                
                attendance_data = load_attendance_data()
                
                if uid_face:
                    # Get status for specific user
                    uid_face = str(uid_face)
                    
                    if uid_face not in attendance_data or date_param not in attendance_data[uid_face]:
                        return {
                            "date": date_param,
                            "total_records": 0,
                            "attendance": []
                        }
                    
                    record = attendance_data[uid_face][date_param]
                    
                    return {
                        "date": date_param,
                        "total_records": 1,
                        "attendance": [{
                            "uid_face": uid_face,
                            "name": record["name"],
                            "date": date_param,
                            "status": record["status"],
                            "clock_in_time": record.get("clock_in_time"),
                            "clock_out_time": record.get("clock_out_time"),
                            "duration": record.get("duration"),
                            "duration_hours": record.get("duration_hours")
                        }]
                    }
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
                    
                    return {
                        "date": date_param,
                        "total_records": len(all_status),
                        "attendance": all_status
                    }
            except Exception as e:
                return {"error": f"Error retrieving attendance status: {str(e)}"}, 500

    return api