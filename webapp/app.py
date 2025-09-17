import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from inference import predict_from_path
import pathlib

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

template_dir = os.path.join(pathlib.Path(__file__).resolve().parent, 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            try:
                res = predict_from_path(save_path)
            except Exception as e:
                return render_template('index.html', error=f'Error during prediction: {e}')
            return render_template('result.html', filename=filename, result=res)
        else:
            return render_template('index.html', error='Unsupported file type')
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
