from flask import Blueprint, request, jsonify
from utils.smoke_detection_model_utils import detect_smoke
from models import ImageRecord
from database.db import db
import os

process_bp = Blueprint('process', __name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@process_bp.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    print("Incoming request:", request.method, request.headers)

    if request.method == 'OPTIONS':
        return '', 204  # Preflight request

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        result = detect_smoke(save_path)
        label = result['result']
        confidence = result['confidence']
        print(f"label: {label}, confidence: {confidence} ({type(confidence)})")



        # Optional DB save
        new_entry = ImageRecord(
            filename=file.filename,
            prediction=label,
            confidence=confidence
        )
        db.session.add(new_entry)
        db.session.commit()

        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'filename': file.filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


