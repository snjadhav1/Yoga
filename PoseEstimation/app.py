from flask import Flask, render_template, request, jsonify, Response
import os
import numpy as np
import cv2
import base64
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# Import the required functions directly
# These were defined in your notebook.ipynb but we'll import them here explicitly
from movenet_utils import (
    evaluate_yoga_pose,
    process_video,
    calculate_angle,
    extract_keypoints,
    keypoints_to_features,
    evaluate_pose_correctness,
    draw_pose
)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model and label encoder
try:
    # Load model
    yoga_model = load_model('yoga_pose_model.keras')
    # Load label encoder
    with open('yoga_label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    print("Model and label encoder loaded successfully")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    # You might want to exit here or handle the error appropriately
    # import sys
    # sys.exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save uploaded file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    
    # Call the evaluate_yoga_pose function
    output_image, pose_class, is_correct, feedback = evaluate_yoga_pose(
        image_path, yoga_model, label_encoder)
    
    if output_image is None:
        return jsonify({'error': 'Could not process image'}), 500
    
    # Convert output image to base64 for display
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate overall score based on feedback
    accuracy_score = calculate_accuracy_score(feedback, is_correct)
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_str}',
        'pose_class': pose_class,
        'is_correct': is_correct,
        'feedback': feedback,
        'accuracy_score': accuracy_score
    })

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Save uploaded file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    
    # Generate output video path
    output_video = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{file.filename}')
    
    # Process video using the function
    process_video(source=video_path, output_path=output_video, 
                 yoga_model=yoga_model, label_encoder=label_encoder)
    
    return jsonify({
        'video_url': f'/static/uploads/output_{file.filename}'
    })

@app.route('/webcam-frame', methods=['POST'])
def process_webcam_frame():
    # Get base64 encoded image from request
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data'}), 400
    
    # Convert base64 to image
    encoded_data = data['frame'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save frame temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
    cv2.imwrite(temp_path, frame)
    
    # Process the frame
    output_image, pose_class, is_correct, feedback = evaluate_yoga_pose(
        temp_path, yoga_model, label_encoder)
    
    if output_image is None:
        return jsonify({'error': 'Could not process image'}), 500
    
    # Convert output image to base64 for display
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate overall score
    accuracy_score = calculate_accuracy_score(feedback, is_correct)
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_str}',
        'pose_class': pose_class,
        'is_correct': is_correct,
        'feedback': feedback,
        'accuracy_score': accuracy_score
    })

def calculate_accuracy_score(feedback, is_correct):
    # Simple scoring based on feedback
    if is_correct:
        return 95  # High score for correct poses
    
    # Count issues
    issues = len([f for f in feedback if "Confidence" not in f and "Cannot evaluate" not in f])
    
    # Deduct points based on issues
    score = max(60, 90 - (issues * 10))
    return score

if __name__ == '__main__':
    app.run(debug=True)