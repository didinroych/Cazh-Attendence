#!/usr/bin/env python
"""
detect.py - Face detection and recognition

This script detects and recognizes faces in images or video streams using
pre-trained face recognition embeddings.

Usage:
  python detect.py --embeddings path/to/embeddings.pkl [--image path/to/image.jpg] [--video path/to/video.mp4] [--webcam]
"""

import os
import cv2
import argparse
import pickle
import numpy as np
from datetime import datetime

class FaceDetectionRecognition:
    """
    Class for face detection and recognition using OpenCV DNN
    """
    def __init__(self, embeddings_path, model_dir="models"):
        """
        Initialize the face detection and recognition system
        
        Args:
            embeddings_path: Path to the embeddings database pickle file
            model_dir: Directory containing the face detection and recognition models
        """
        # Setup paths
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Model paths
        self.fd_model_path = os.path.join(self.model_dir, "face_detection_yunet.onnx")
        self.fr_model_path = os.path.join(self.model_dir, "face_recognition_sface.onnx")
        
        # Check if models exist, otherwise prompt for download
        self.check_models()
        
        # Configure detection parameters
        self.conf_threshold = 0.9  # Confidence threshold for face detection
        self.nms_threshold = 0.3   # Non-maximum suppression threshold
        self.top_k = 5000          # Keep top K detections
        self.recognition_threshold = 0.5  # Threshold for face similarity (lower is stricter)
        
        # Initialize face detector and recognizer
        self.init_face_detector()
        self.init_face_recognizer()
        
        # Load embeddings database
        self.load_embeddings(embeddings_path)
    
    def check_models(self):
        """Check if required models exist, otherwise prompt for download"""
        models_missing = False
        
        if not os.path.exists(self.fd_model_path):
            models_missing = True
            print(f"Face detection model missing: {self.fd_model_path}")
        
        if not os.path.exists(self.fr_model_path):
            models_missing = True
            print(f"Face recognition model missing: {self.fr_model_path}")
        
        if models_missing:
            print("\nPlease download the following models from OpenCV Model Zoo:")
            print("1. Face Detection Model (face_detection_yunet.onnx):")
            print("   https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")
            print("\n2. Face Recognition Model (face_recognition_sface.onnx):")
            print("   https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface")
            print(f"\nPlace them in the '{self.model_dir}' directory.")
            
            choice = input("\nDo you want to continue without downloading (may fail)? (y/n): ")
            if choice.lower() != 'y':
                exit(1)
    
    def init_face_detector(self):
        """Initialize the face detector model"""
        if not os.path.exists(self.fd_model_path):
            raise FileNotFoundError(f"Face detection model not found at {self.fd_model_path}")
        
        # Initialize DNN Face Detector
        self.face_detector = cv2.FaceDetectorYN.create(
            model=self.fd_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )
    
    def init_face_recognizer(self):
        """Initialize the face recognizer model"""
        if not os.path.exists(self.fr_model_path):
            raise FileNotFoundError(f"Face recognition model not found at {self.fr_model_path}")
        
        # Initialize DNN Face Recognizer
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            model=self.fr_model_path,
            config=""
        )
    
    # def load_embeddings(self, embeddings_path):
    #     """
    #     Load face embeddings from a pickle file
        
    #     Args:
    #         embeddings_path: Path to the embeddings database pickle file
    #     """
    #     if not os.path.exists(embeddings_path):
    #         raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
        
    #     with open(embeddings_path, 'rb') as f:
    #         self.embeddings_db = pickle.load(f)
        
    #     print(f"Loaded {len(self.embeddings_db)} face profiles")

    def load_embeddings(self, embeddings_path):
        """
        Load face embeddings from a pickle file safely.
        If the file doesn't exist or is empty, initialize an empty database.
        """
        self.embeddings_db = {}  # Default: empty DB

        if not os.path.exists(embeddings_path):
            print(f"[INFO] Embeddings file not found at '{embeddings_path}', starting fresh.")
            return

        if os.path.getsize(embeddings_path) == 0:
            print(f"[INFO] Embeddings file '{embeddings_path}' is empty, starting fresh.")
            return

        try:
            with open(embeddings_path, 'rb') as f:
                self.embeddings_db = pickle.load(f)
            print(f"[INFO] Loaded {len(self.embeddings_db)} face profiles from '{embeddings_path}'")
        except Exception as e:
            print(f"[ERROR] Failed to load embeddings from '{embeddings_path}': {e}")
            print("[INFO] Starting with an empty database.")
            self.embeddings_db = {}

    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            faces: Detected face locations
        """
        # Get frame dimensions
        height, width, _ = frame.shape
        
        # Set input size for face detector
        self.face_detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = self.face_detector.detect(frame)
        
        return faces
    
    def get_face_embedding(self, frame, face):
        """
        Extract face embedding from detected face
        
        Args:
            frame: Input image frame
            face: Face coordinates [x, y, width, height, confidence]
            
        Returns:
            Face embedding vector (feature vector)
        """
        # Align face
        aligned_face = self.face_recognizer.alignCrop(frame, face)
        
        # Get face feature (embedding)
        face_feature = self.face_recognizer.feature(aligned_face)
        
        return face_feature
    """
    Update the recognize_face function to use manual cosine similarity 
    instead of the cv2.FaceRecognizerSF.FR_COSINE constant
    """

    def recognize_face(self, frame, face):
        """
        Recognize a face from the database
        
        Args:
            frame: Input image frame
            face: Face coordinates
            
        Returns:
            name: Recognized person name or "Unknown"
            confidence: Recognition confidence score
        """

        aligned_face = self.face_recognizer.alignCrop(frame, face)
        if self.is_spoof_frame(aligned_face):
            return "Spoof Detected", 0.0
        if len(self.embeddings_db) == 0:
            return "Unknown", 0.0
        
        try:
            # Get face embedding
            face_feature = self.get_face_embedding(frame, face)
            
            # Compare with database
            best_match = None
            best_score = 0.0
            
            for name, stored_feature in self.embeddings_db.items():
                # Calculate cosine similarity manually
                # Normalize vectors to unit length
                norm_face = np.linalg.norm(face_feature)
                norm_stored = np.linalg.norm(stored_feature)
                
                if norm_face > 0 and norm_stored > 0:
                    # Calculate cosine similarity: dot product of normalized vectors
                    score = np.dot(face_feature, stored_feature.T) / (norm_face * norm_stored)
                    # Convert to scalar value
                    score = float(score)
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_match = name
            
            # Check if score is above threshold
            if best_score > self.recognition_threshold:
                return best_match, best_score
            else:
                return "Unknown", best_score
                
        except Exception as e:
            print(f"Error recognizing face: {str(e)}")
            return "Error", 0.0

    def process_frame(self, frame):
        """
        Process a frame for face detection and recognition
        
        Args:
            frame: Input image frame
            
        Returns:
            processed_frame: Frame with annotations
            faces_info: Information about detected faces
        """
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        faces_info = []
        
        # Process each detected face
        if faces is not None:
            for face in faces:
                # Extract face coordinates
                x, y, w, h, conf = int(face[0]), int(face[1]), int(face[2]), int(face[3]), face[-1]
                
                # Recognize the face
                name, score = self.recognize_face(frame, face)
                label = f"{name} ({score:.2f})"
                
                # Color based on recognition status
                if name == "Unknown":
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (0, 255, 0)  # Green for recognized
                
                # Draw bounding box and label
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add to faces info
                faces_info.append({
                    "bbox": (x, y, w, h),
                    "confidence": conf,
                    "name": name,
                    "score": score
                })
        
        return display_frame, faces_info
    
    def process_image(self, image_path, output_path=None):
        """
        Process a single image for face detection and recognition
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output image with annotations
            
        Returns:
            processed_image: Image with face annotations
            faces_info: Information about detected faces
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image: {image_path}")
            return None, []
        
        # Process frame
        processed_frame, faces_info = self.process_frame(frame)
        
        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"Saved output to {output_path}")
        
        return processed_frame, faces_info
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process a video for face detection and recognition
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video with annotations
            display: Whether to display the processed video in a window
            
        Returns:
            success: True if processing was successful
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, faces_info = self.process_frame(frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Face Recognition', processed_frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write frame if output path provided
                if output_path:
                    writer.write(processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            print(f"Processed {frame_count} frames in total")
            
        finally:
            # Release resources
            cap.release()
            if output_path:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        return True
    
    def process_webcam(self, camera_id=0, output_dir=None):
        """
        Process webcam feed for face detection and recognition
        
        Args:
            camera_id: ID of the webcam to use
            output_dir: Directory to save captured images
            
        Returns:
            success: True if processing was successful
        """
        # Create output directory if provided
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Could not open webcam with ID {camera_id}")
            return False
        
        print("\n=== Face Recognition Webcam Mode ===")
        print("Controls:")
        print("  q - Quit")
        print("  s - Save current frame")
        print("  + - Increase recognition threshold")
        print("  - - Decrease recognition threshold")
        
        frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, faces_info = self.process_frame(frame)
                
                # Add info text
                threshold_text = f"Recognition Threshold: {self.recognition_threshold:.2f}"
                cv2.putText(processed_frame, threshold_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Face Recognition', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Exit on 'q' key
                if key == ord('q'):
                    break
                
                # Save frame on 's' key
                elif key == ord('s') and output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create filename with detected face information
                    if faces_info:
                        face_names = "_".join([face["name"] for face in faces_info])
                        filename = f"{output_dir}/capture_{timestamp}_{face_names}.jpg"
                    else:
                        filename = f"{output_dir}/capture_{timestamp}_no_faces.jpg"
                    
                    cv2.imwrite(filename, processed_frame)
                    print(f"Saved frame to {filename}")
                
                # Adjust recognition threshold
                elif key == ord('+') or key == ord('='):
                    self.recognition_threshold = min(1.0, self.recognition_threshold + 0.05)
                    print(f"Recognition threshold increased to {self.recognition_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.recognition_threshold = max(0.0, self.recognition_threshold - 0.05)
                    print(f"Recognition threshold decreased to {self.recognition_threshold:.2f}")
                
                frame_count += 1
            
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
        
        return True
    def is_spoof_frame(self, aligned_face, blur_thresh=100.0):
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score < blur_thresh  # True = Spoof (terlalu blur)



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face detection and recognition")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings file")
    parser.add_argument("--models", default="models", help="Path to models directory")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID for webcam")
    parser.add_argument("--output", help="Path to output file/directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Recognition threshold")
    args = parser.parse_args()
    
    # Create detector
    detector = FaceDetectionRecognition(args.embeddings, model_dir=args.models)
    
    # Set recognition threshold if provided
    if args.threshold:
        detector.recognition_threshold = args.threshold
    
    # Process input based on arguments
    if args.image:
        # Process image
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
        
        # Determine output path
        output_path = args.output
        if not output_path:
            filename, ext = os.path.splitext(os.path.basename(args.image))
            output_path = f"{filename}_detected{ext}"
        
        # Process image
        processed_image, faces_info = detector.process_image(args.image, output_path)
        
        # Display results
        if processed_image is not None:
            cv2.imshow("Face Recognition", processed_image)
            print(f"Detected {len(faces_info)} faces")
            for i, face in enumerate(faces_info):
                print(f"  Face {i+1}: {face['name']} (confidence: {face['score']:.2f})")
            
            # Wait for key press
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.video:
        # Process video
        if not os.path.exists(args.video):
            print(f"Video file not found: {args.video}")
            return
        
        # Determine output path
        output_path = args.output
        if not output_path:
            filename, ext = os.path.splitext(os.path.basename(args.video))
            output_path = f"{filename}_detected.mp4"
        
        # Process video
        detector.process_video(args.video, output_path)
    
    elif args.webcam:
        # Process webcam feed
        detector.process_webcam(args.camera_id, args.output)
    
    else:
        print("Please specify an input source (--image, --video, or --webcam)")


if __name__ == "__main__":
    main()