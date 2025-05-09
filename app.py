from flask import Flask, request, jsonify
from google.cloud import vision
import base64
import os

app = Flask(__name__)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "visionapi.json"
client = vision.ImageAnnotatorClient()

@app.route('/api/ocr', methods=['POST'])
def ocr():
    try:
        data = request.get_json()
        image_content = base64.b64decode(data['image'])
        image = vision.Image(content=image_content)

        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            return jsonify({'text': texts[0].description})
        else:
            return jsonify({'text': ''})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
