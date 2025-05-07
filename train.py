#!/usr/bin/env python
"""
train.py - Train a face recognition model using existing images

This script processes images in a directory structure where each subdirectory 
represents a person and contains their face images. It extracts face embeddings 
and saves them to a database file for later use in face recognition.

Usage:
  python train.py --dataset path/to/dataset --output path/to/output

Directory structure:
  dataset/
  ├── person1/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── person2/
  │   ├── image1.jpg
  │   └── image2.jpg
"""

import os
import cv2
import argparse
import pickle
import numpy as np
from tqdm import tqdm

class FaceRecognitionTrainer:
    """
    Class for training a face recognition model using existing images
    """
    def __init__(self, model_dir="models"):
        """
        Initialize the face recognition trainer

        Args:
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
        
        # Initialize face detector and recognizer
        self.init_face_detector()
        self.init_face_recognizer()
        
        # Storage for face embeddings
        self.embeddings_db = {}
    
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
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: Input image
            
        Returns:
            faces: Detected face locations
        """
        # Get frame dimensions
        height, width, _ = image.shape
        
        # Set input size for face detector
        self.face_detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = self.face_detector.detect(image)
        
        return faces
    
    def get_face_embedding(self, image, face):
        """
        Extract face embedding from detected face
        
        Args:
            image: Input image
            face: Face coordinates [x, y, width, height, confidence]
            
        Returns:
            Face embedding vector (feature vector)
        """
        # Extract face coordinates
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        
        # Align face
        aligned_face = self.face_recognizer.alignCrop(image, face)
        
        # Get face feature (embedding)
        face_feature = self.face_recognizer.feature(aligned_face)
        
        return face_feature
    
    def process_image(self, image_path, person_name):
        """
        Process a single image to extract face embedding
        
        Args:
            image_path: Path to the image file
            person_name: Name of the person in the image
            
        Returns:
            success: True if face was successfully processed
            embedding: Face embedding if successful, None otherwise
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return False, None
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if faces is None or len(faces) == 0:
                print(f"No faces found in {image_path}")
                return False, None
            
            # If multiple faces found, use the one with highest confidence
            if len(faces) > 1:
                # Sort by confidence (5th column)
                face_idx = np.argmax(faces[:, 4])
                face = faces[face_idx]
                print(f"Multiple faces found in {image_path}. Using face with highest confidence.")
            else:
                face = faces[0]
            
            # Get face embedding
            embedding = self.get_face_embedding(image, face)
            
            return True, embedding
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False, None
    
    def train_from_directory(self, dataset_dir, output_path, min_images=1):
        """
        Process all images in the dataset directory to create face embeddings
        
        Args:
            dataset_dir: Path to the dataset directory containing person subdirectories
            output_path: Path to save the embeddings database
            min_images: Minimum number of images required for a person to be included
            
        Returns:
            success: True if training was successful
        """
        # Store embeddings for each person
        embeddings_db = {}
        person_count = 0
        image_count = 0
        
        # Get list of person directories
        try:
            person_dirs = [d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d))]
        except Exception as e:
            print(f"Error accessing dataset directory {dataset_dir}: {str(e)}")
            return False
        
        if not person_dirs:
            print(f"No person directories found in {dataset_dir}")
            return False
        
        print(f"Found {len(person_dirs)} person directories in the dataset.")
        
        # Process each person directory
        for person_dir in tqdm(person_dirs, desc="Processing persons"):
            person_name = person_dir
            person_path = os.path.join(dataset_dir, person_dir)
            
            # Get image files for this person
            image_files = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(image_files) < min_images:
                print(f"Skipping {person_name}: Not enough images ({len(image_files)} < {min_images})")
                continue
            
            print(f"\nProcessing {person_name} with {len(image_files)} images...")
            
            # Process each image for this person
            successful_embeddings = []
            
            for image_file in tqdm(image_files, desc=f"Processing {person_name}", leave=False):
                image_path = os.path.join(person_path, image_file)
                success, embedding = self.process_image(image_path, person_name)
                
                if success:
                    successful_embeddings.append(embedding)
                    image_count += 1
            
            # Only add person if we have at least one successful embedding
            if successful_embeddings:
                # Calculate average embedding for this person
                if len(successful_embeddings) > 1:
                    # Convert list of embeddings to numpy array
                    embeddings_array = np.array(successful_embeddings)
                    # Average the embeddings
                    avg_embedding = np.mean(embeddings_array, axis=0)
                    # Normalize to unit length
                    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                    embeddings_db[person_name] = avg_embedding
                else:
                    # Just use the single embedding
                    embeddings_db[person_name] = successful_embeddings[0]
                
                person_count += 1
                print(f"Added {person_name} with {len(successful_embeddings)} face embeddings.")
            else:
                print(f"Failed to extract any valid embeddings for {person_name}.")
        
        # Save embeddings database to file
        if person_count > 0:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings_db, f)
            
            print(f"\nTraining complete! Processed {image_count} images for {person_count} persons.")
            print(f"Face embeddings saved to {output_path}")
            return True
        else:
            print("No valid embeddings were extracted. Training failed.")
            return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train face recognition model from dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", default="embeddings.pkl", help="Path to output embeddings file")
    parser.add_argument("--models", default="models", help="Path to models directory")
    parser.add_argument("--min-images", type=int, default=1, help="Minimum images per person")
    args = parser.parse_args()
    
    # Check if dataset directory exists
    if not os.path.exists(args.dataset):
        print(f"Dataset directory not found: {args.dataset}")
        return
    
    # Create trainer
    trainer = FaceRecognitionTrainer(model_dir=args.models)
    
    # Train from directory
    trainer.train_from_directory(args.dataset, args.output, args.min_images)


if __name__ == "__main__":
    main()