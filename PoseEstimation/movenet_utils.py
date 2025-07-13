import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import pickle
from tensorflow.keras.models import load_model

# Force CPU usage for simplicity
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Define keypoint names
KEYPOINTS = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

# Yoga pose correctness criteria
POSE_CRITERIA = {
    'downdog': [
        {'description': 'Arms should be straight', 'joints': [(5, 7, 9), (6, 8, 10)], 'angle_range': (160, 180)},
        {'description': 'Body should form an inverted V', 'joints': [(5, 11, 13), (6, 12, 14)], 'angle_range': (90, 130)}
    ],
    'goddess': [
        {'description': 'Knees should be bent at ~90°', 'joints': [(11, 13, 15), (12, 14, 16)], 'angle_range': (85, 115)},
        {'description': 'Arms should be at shoulder level', 'joints': [(3, 5, 7), (4, 6, 8)], 'angle_range': (70, 110)}
    ],
    'plank': [
        {'description': 'Body should be straight', 'joints': [(5, 11, 15), (6, 12, 16)], 'angle_range': (160, 180)},
        {'description': 'Arms should be perpendicular to ground', 'joints': [(5, 7, 9), (6, 8, 10)], 'angle_range': (75, 105)}
    ],
    'tree': [
        {'description': 'Standing leg should be straight', 'joints': [(12, 14, 16)], 'angle_range': (160, 180)},
        {'description': 'Raised foot should be against thigh', 'joints': [(11, 13, 15)], 'angle_range': (10, 60)}
    ],
    'warrior2': [
        {'description': 'Front knee should be bent at ~90°', 'joints': [(11, 13, 15)], 'angle_range': (85, 115)},
        {'description': 'Back leg should be straight', 'joints': [(12, 14, 16)], 'angle_range': (160, 180)},
        {'description': 'Arms should be parallel to ground', 'joints': [(5, 7, 9), (6, 8, 10)], 'angle_range': (160, 180)}
    ]
}

def process_image(image_path, image_size=192):
    """Read image, preprocess it for MoveNet"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None, None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    height, width, _ = image.shape
    
    # Resize and pad image
    input_image = tf.image.resize_with_pad(tf.convert_to_tensor(image), image_size, image_size)
    input_image = tf.cast(tf.expand_dims(input_image, axis=0), dtype=tf.int32)
    
    return image, input_image, (height, width)

def draw_pose(image, keypoints, height, width, pose_class=None, feedback=None, threshold=0.3):
    """Draw pose keypoints and skeleton on the image"""
    output_image = image.copy()
    
    # Define connections for skeleton
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 6), (5, 11), (6, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Draw keypoints
    for idx, (y, x, confidence) in enumerate(keypoints):
        if confidence > threshold:
            x_px = int(x * width)
            y_px = int(y * height)
            
            # Draw circle
            cv2.circle(output_image, (x_px, y_px), 5, (0, 255, 0), -1)
            
            # Label keypoint
            cv2.putText(output_image, f"{KEYPOINTS[idx]}", (x_px + 5, y_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        
        y1, x1, conf1 = keypoints[start_idx]
        y2, x2, conf2 = keypoints[end_idx]
        
        if conf1 > threshold and conf2 > threshold:
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            
            cv2.line(output_image, (x1_px, y1_px), (x2_px, y2_px), (255, 0, 0), 2)
    
    # Display pose class and feedback
    if pose_class:
        cv2.putText(output_image, f"Pose: {pose_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if feedback:
        y_pos = 60
        for line in feedback:
            cv2.putText(output_image, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_pos += 25
    
    return output_image

def calculate_angle(a, b, c):
    """Calculate angle between three points (in radians)"""
    a = np.array([a[1], a[0]])  # Convert y,x to x,y
    b = np.array([b[1], b[0]])
    c = np.array([c[1], c[0]])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def extract_keypoints(image_path):
    """Extract keypoints from an image using MoveNet"""
    original_image, input_image, dimensions = process_image(image_path)
    if original_image is None:
        return None, None, None
    
    # Run inference
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0][0]
    
    return keypoints, original_image, dimensions

def keypoints_to_features(keypoints, conf_threshold=0.3):
    """Convert keypoints to a flattened feature vector"""
    # Filter low confidence keypoints
    keypoints_copy = keypoints.copy()  # Create a copy to avoid modifying the original
    for i in range(len(keypoints_copy)):
        if keypoints_copy[i][2] < conf_threshold:
            keypoints_copy[i] = [0, 0, 0]  # Set low confidence keypoints to origin
    
    # Flatten the keypoints (y, x, conf) into a single vector
    features = keypoints_copy.flatten()
    
    # Add derived features - pairwise distances between keypoints
    for i in range(17):
        for j in range(i+1, 17):
            # Only if both keypoints have good confidence
            if keypoints_copy[i][2] > conf_threshold and keypoints_copy[j][2] > conf_threshold:
                # Calculate Euclidean distance between points
                dist = np.sqrt((keypoints_copy[i][0] - keypoints_copy[j][0])**2 + 
                              (keypoints_copy[i][1] - keypoints_copy[j][1])**2)
                features = np.append(features, dist)
            else:
                features = np.append(features, 0)
    
    return features

def evaluate_pose_correctness(keypoints, pose_class):
    """Evaluate if the pose is correct based on pose-specific criteria"""
    feedback = []
    correct = True
    
    if pose_class not in POSE_CRITERIA:
        return True, ["No specific criteria defined for this pose"]
    
    criteria = POSE_CRITERIA[pose_class]
    
    for criterion in criteria:
        for joint_set in criterion['joints']:
            # Skip if any keypoint has low confidence
            if any(keypoints[joint][2] < 0.3 for joint in joint_set):
                feedback.append(f"Cannot evaluate: {criterion['description']} - joints not visible")
                continue
                
            # Calculate angle
            angle = calculate_angle(
                keypoints[joint_set[0]], 
                keypoints[joint_set[1]], 
                keypoints[joint_set[2]]
            )
            
            min_angle, max_angle = criterion['angle_range']
            if min_angle <= angle <= max_angle:
                # This criterion is satisfied
                pass
            else:
                correct = False
                if angle < min_angle:
                    feedback.append(f"{criterion['description']} - angle too small ({angle:.1f}°)")
                else:
                    feedback.append(f"{criterion['description']} - angle too large ({angle:.1f}°)")
    
    if correct and not feedback:
        feedback.append("Pose is correct! Great job!")
        
    return correct, feedback

def evaluate_yoga_pose(image_path, yoga_model, label_encoder):
    """Evaluate a yoga pose in an image"""
    # Extract keypoints
    keypoints, original_image, dimensions = extract_keypoints(image_path)
    if keypoints is None or original_image is None:
        return None, None, None, None
    
    height, width = dimensions
    
    # Convert keypoints to features
    features = keypoints_to_features(keypoints)
    features = features.reshape(1, -1)  # Reshape for model input
    
    # Predict pose class
    prediction = yoga_model.predict(features, verbose=0)[0]
    pose_class_idx = np.argmax(prediction)
    pose_class = label_encoder.inverse_transform([pose_class_idx])[0]
    confidence = prediction[pose_class_idx]
    
    # Evaluate pose correctness
    is_correct, feedback = evaluate_pose_correctness(keypoints, pose_class)
    
    # Add confidence to feedback
    feedback.insert(0, f"Confidence: {confidence:.2f}")
    
    # Draw pose with feedback
    output_image = draw_pose(original_image, keypoints, height, width, 
                           pose_class=pose_class, feedback=feedback)
    
    return output_image, pose_class, is_correct, feedback

def process_video(source=0, output_path=None, yoga_model=None, label_encoder=None):
    """Process video from file or webcam (source=0 for webcam)"""
    # If models aren't provided, try to load them
    if yoga_model is None or label_encoder is None:
        try:
            yoga_model = load_model('yoga_pose_model.keras')
            with open('yoga_label_encoder.pkl', 'rb') as file:
                label_encoder = pickle.load(file)
        except Exception as e:
            print(f"Error loading model or label encoder: {e}")
            return
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad image
        input_image = tf.image.resize_with_pad(tf.convert_to_tensor(rgb_frame), 192, 192)
        input_image = tf.cast(tf.expand_dims(input_image, axis=0), dtype=tf.int32)
        
        # Run inference
        outputs = movenet(input_image)
        keypoints = outputs['output_0'].numpy()[0][0]
        
        # Convert keypoints to features
        features = keypoints_to_features(keypoints)
        features = features.reshape(1, -1)
        
        # Predict pose class
        prediction = yoga_model.predict(features, verbose=0)[0]
        pose_class_idx = np.argmax(prediction)
        pose_class = label_encoder.inverse_transform([pose_class_idx])[0]
        confidence = prediction[pose_class_idx]
        
        # Evaluate pose correctness
        is_correct, feedback = evaluate_pose_correctness(keypoints, pose_class)
        
        # Add confidence to feedback
        feedback.insert(0, f"Confidence: {confidence:.2f}")
        
        # Draw pose with feedback
        output_frame = draw_pose(rgb_frame, keypoints, frame_height, frame_width, 
                                pose_class=pose_class, feedback=feedback)
        
        # Convert back to BGR for displaying with OpenCV
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        # Display the resulting frame
        cv2.imshow('Yoga Pose Evaluation', output_frame)
        
        # Write frame to output video if specified
        if writer:
            writer.write(output_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()