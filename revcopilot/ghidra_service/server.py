from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/decompile', methods=['POST'])
def decompile():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join('/tmp', file.filename)
    file.save(temp_file_path)

    # Call Ghidra headless decompilation command
    try:
        output = subprocess.check_output(['ghidraRun', 'decompile', temp_file_path], stderr=subprocess.STDOUT)
        return jsonify({'output': output.decode('utf-8')}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.output.decode('utf-8')}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)