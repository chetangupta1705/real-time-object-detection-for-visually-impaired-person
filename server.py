from flask import Flask, jsonify
import threading
import time

app = Flask(__name__)
latest_detection = {"object": "none", "distance": 0}

@app.route("/result")
def result():
    return jsonify(latest_detection)

def update_detection(obj, dist):
    global latest_detection
    latest_detection = {"object": obj, "distance": dist, "timestamp": time.time()}
