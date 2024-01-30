from flask import Flask, render_template
from flask_socketio import SocketIO
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

image_folder = "static/image"
current_image_index = 0

@app.route("/")
def home():
    return render_template('index_simple.html')

@socketio.on('request_new_image')
def handle_request_new_image(data):
    global current_image_index

    # # Use the provided index or default to 0 if not provided
    index = int(data.get('index', 0))
    # # Validate the index to stay within the range of available images
    # image_files = os.listdir(image_folder)
    # if 0 <= index < len(image_files):
    current_image_index = index
    # else:
    #     current_image_index = 0
    dynamic_image_file = get_dynamic_image()
    socketio.emit('new_image', {'image_file': dynamic_image_file})
    print('send finish')

def get_dynamic_image():
    image_files = os.listdir(image_folder)
    image_files.sort()
    return f"image/{image_files[current_image_index]}"

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=6006)
