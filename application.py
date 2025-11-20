from flask import Flask, render_template, jsonify
from weather1 import fetch_and_process_forecast

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/weather")
def weather():
    df = fetch_and_process_forecast()
    if df is not None and len(df) > 0:
        return jsonify(df.iloc[0].to_dict())
    return jsonify({"error": "No data"}), 500

if __name__ == "__main__":

    app.run(debug=True)
