from flask_cors import CORS
from database.db import db
from database.config import Config
from routes.process import process_bp
import logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, request, send_file
from utils.cyclegan_utils import generate_enhanced_image
import io
import os


app = Flask(__name__)
CORS(app)  # 👈 This line enables CORS for all routes

app.config.from_object(Config)
db.init_app(app)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:5173"}})

# Register Routes
app.register_blueprint(process_bp)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(port=5000, debug=True)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
            
        file = request.files['image']
        if file.filename == '':
            return {'error': 'Empty filename'}, 400
            
        # Save temp file (optional)
        temp_path = os.path.join('uploads', file.filename)
        file.save(temp_path)
        
        # Process image
        output_img = generate_enhanced_image(temp_path)
        
        # Convert to bytes
        img_io = io.BytesIO()
        output_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        return {'error': str(e)}, 500