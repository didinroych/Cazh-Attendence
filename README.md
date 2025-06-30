# 👤 Face Recognition Attendance System

A comprehensive face recognition-based attendance management system built with Flask, OpenCV, and modern web technologies. Features real-time face detection, anti-spoofing, and a complete REST API with Swagger documentation.

## 🚀 Features

### 🎯 Core Functionality
- **Real-time Face Detection** using OpenCV DNN models
- **Face Recognition** with high accuracy
- **Anti-Spoofing Detection** to prevent fake face attacks
- **Attendance Management** with clock in/out functionality
- **User Registration** via video recording
- **Live Camera Verification** with auto-verification mode

### 🌐 Web Interface
- **Responsive Design** that works on desktop and mobile
- **Live Camera Feed** for real-time face verification
- **User Registration Page** with video capture
- **Attendance Dashboard** with real-time updates
- **User Management** interface

### 🔧 API Features
- **RESTful API** with comprehensive endpoints
- **Swagger Documentation** (interactive API docs)
- **JSON Response Format** for easy integration
- **Error Handling** with detailed messages
- **File Upload Support** for images and videos

## 📱 Screenshots

### Web Interface
- **Live Camera**: Real-time face verification with confidence scores
- **Registration**: Video-based face enrollment system
- **Attendance**: Clock in/out with face verification
- **Dashboard**: View attendance records and user management

### API Documentation
- **Swagger UI**: Interactive API testing at `/api-docs/`
- **Organized Endpoints**: Face recognition, attendance, user management
- **Try It Out**: Test endpoints directly from browser

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/face-recognition-attendance.git
cd face-recognition-attendance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Face Recognition Models
Download the required models from OpenCV Model Zoo:

**Face Detection Model:**
```bash
wget https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
mv face_detection_yunet_2023mar.onnx models/face_detection_yunet.onnx
```

**Face Recognition Model:**
```bash
wget https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
mv face_recognition_sface_2021dec.onnx models/face_recognition_sface.onnx
```

Or create the `models/` directory and manually download:
- [Face Detection Model](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
- [Face Recognition Model](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface)

### 4. Run Application
```bash
python app.py
```

### 5. Access Application
- **Web Interface**: http://localhost:5000
- **API Documentation**: http://localhost:5000/api-docs/

## 📁 Project Structure

```
face-recognition-attendance/
├── app.py                      # Main Flask application
├── detect.py                   # Face detection & recognition class
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── api/                        # API configuration
│   ├── __init__.py
│   └── swagger_config.py       # Swagger/API documentation
├── models/                     # Face recognition models
│   ├── face_detection_yunet.onnx
│   └── face_recognition_sface.onnx
├── static/                     # CSS & JavaScript
│   ├── css/
│   │   └── styles.css          # Shared styles
│   └── js/
│       ├── live-cam.js         # Live camera functionality
│       ├── attendance.js       # Attendance management 
│       └── user-list.js        # User management
├── templates/                  # HTML templates
│   ├── live-cam.html          # Live camera page
│   ├── absensi.html           # Attendance page
│   ├── capture.html           # Registration page
│   └── user-list.html         # User list page
├── video/                      # Uploaded registration videos
├── embeddings.pkl             # Face embeddings database
└── attendance.json            # Attendance records
```

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_DEBUG=True          # Enable debug mode
export FLASK_HOST=0.0.0.0       # Host address
export FLASK_PORT=5000          # Port number
```

### Face Recognition Settings
Edit `detect.py` to adjust:
```python
self.conf_threshold = 0.9        # Face detection confidence
self.recognition_threshold = 0.5  # Face recognition threshold
self.nms_threshold = 0.3         # Non-maximum suppression
```

## 🌐 API Documentation

### Base URL
- **Development**: `http://localhost:5000`
- **API Prefix**: `/api`
- **Swagger UI**: `/api-docs/`

### Main Endpoints

#### Face Recognition
- `POST /api/face/verify` - Verify face from image
- `POST /api/face/register` - Register new face (best frame)
- `POST /api/face/register-average` - Register new face (average embedding)

#### Attendance Management  
- `POST /api/attendance/clock-in` - Clock in with face verification
- `POST /api/attendance/clock-out` - Clock out with face verification
- `GET /api/attendance/status` - Get attendance status

#### User Management
- `GET /api/users/list` - Get all registered users

### Legacy Endpoints
For backward compatibility, original endpoints are maintained:
- `POST /verify-face`
- `POST /clock-in`
- `POST /clock-out` 
- `GET /attendance-status`
- `GET /get-user-face`

## 📖 Usage Examples

### Web Interface

#### 1. Register New User
1. Go to **Register Face** page
2. Enter user name
3. Click **Record & Register** 
4. Record 5-second video
5. System processes video and creates face profile

#### 2. Live Verification
1. Go to **Live Camera** page
2. Click **Start Camera**
3. Enable **Auto-Verification** for continuous checking
4. Face detection shows confidence scores and identity

#### 3. Attendance Management
1. Go to **Attendance** page  
2. Click **Start Camera**
3. Click **Clock In** to start work day
4. Click **Clock Out** to end work day
5. View real-time attendance list

### API Usage

#### Face Verification
```bash
curl -X POST http://localhost:5000/api/face/verify \
  -F "image=@person.jpg"
```

#### User Registration
```bash
curl -X POST http://localhost:5000/api/face/register-average \
  -F "video=@registration.mp4" \
  -F "name=John Doe"
```

#### Clock In
```bash
curl -X POST http://localhost:5000/api/attendance/clock-in \
  -F "image=@employee.jpg"
```

#### Get Attendance Status
```bash
curl -X GET "http://localhost:5000/api/attendance/status?date=2025-06-30"
```

## 🔒 Security Features

### Anti-Spoofing
- **Blur Detection**: Prevents photo/screen spoofing
- **Confidence Thresholds**: Configurable recognition sensitivity 
 

## 🚀 Deployment

### Development
```bash
python app.py
```

### Error Messages

| Error | Cause | Solution |
|-------|--------|----------|
| `No image uploaded` | Missing file in request | Include image file |
| `Face not recognized` | Unknown face | Register user first |
| `Already clocked in today` | Duplicate clock-in | Check attendance status |
| `No clock in record found` | Clock-out without clock-in | Clock in first |
| `Face verification failed - possible spoof` | Low confidence score | Use live face, improve lighting |


## 🚀 Quick Start Summary

```bash
# 1. Clone & Install
git clone https://github.com/yourusername/face-recognition-attendance.git
cd face-recognition-attendance
pip install -r requirements.txt

# 2. Download Models (place in models/ directory)
# - face_detection_yunet.onnx
# - face_recognition_sface.onnx

# 3. Run Application
python app.py

# 4. Access
# Web: http://localhost:5000
# API: http://localhost:5000/api-docs/
```