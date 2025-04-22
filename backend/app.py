from flask import Flask
from flask_cors import CORS
from database.db import db
from database.config import Config
from routes.process import process_bp
import logging
logging.basicConfig(level=logging.DEBUG)

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
