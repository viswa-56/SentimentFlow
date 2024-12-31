import logging
from flask import Flask, request, jsonify,render_template
import output
from tweets import scrape_tweets

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def hello():
    logging.info("Rendering index1.html")
    return render_template("index1.html")


@app.route('/about')
def about():
    logging.info("Rendering about.html")
    return render_template("about.html")


@app.route('/predict', methods=['POST'])
def predict_house_price():

    input_text = request.form['input']  # Location should remain as a string
    logging.info(f"Received input for prediction: {input_text}")
    predicted_label,val = output.predict(input_text)
    logging.info(f"Predicted label: {predicted_label}, Value: {val}")
    return render_template('result.html', input_text=input_text, predicted_label=predicted_label,val=val)

@app.route('/analyze')
def analyze():
    logging.info("Rendering index.html")
    return render_template('index.html')

@app.route('/predict_twitter', methods=['POST'])
def predict_twitter_sentiment():
    input_text = request.form['input']
    logging.info(f"Received input for Twitter sentiment analysis: {input_text}")
    # query = f"({input_text})"
    tweets, overall_sentiment = scrape_tweets(input_text)
    logging.info(f"Overall sentiment: {overall_sentiment}")
    return render_template('result1.html', overall_sentiment=overall_sentiment,input_text=input_text)

if __name__ == "__main__":
    logging.info("Starting python flask server for house price predictions...")
    print("Starting python flask server for house price predictions...")
    app.run(debug=True, port=9000)
