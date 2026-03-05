from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

LOG_FILE = "attention_log.json"

def _default_data():
    return {
        "total_people_seen": 0,
        "active_now": 0,
        "aisle_stats": {
            "Aisle 1": {"unique_people": 0, "glance": 0, "browse": 0, "dwell": 0},
            "Aisle 2": {"unique_people": 0, "glance": 0, "browse": 0, "dwell": 0},
            "Aisle 3": {"unique_people": 0, "glance": 0, "browse": 0, "dwell": 0},
        },
    }


def load_data():
    if not os.path.exists(LOG_FILE):
        return _default_data()
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return _default_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    return jsonify(load_data())

if __name__ == "__main__":
    print("Dashboard running at http://localhost:5000")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=False, port=5000)