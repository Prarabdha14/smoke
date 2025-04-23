from flask import Blueprint, request, jsonify, send_file
from utils.cyclegan_utils import generate_enhanced_image
import os
import io

process_bp = Blueprint('process', __name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@process_bp.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        return '', 204

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        output_image = generate_enhanced_image(save_path)
        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)



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
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)