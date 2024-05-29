from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
pipe_lr = joblib.load(open("emotion_detection.pkl", "rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the form
        raw_text = request.form['text']
        
        # Make predictions
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        # Get the emotion labels
        labels = pipe_lr.classes_

        return render_template(
            'index.html', 
            text=raw_text, 
            prediction=prediction, 
            probability=zip(labels, probability[0])
        )

if __name__ == '__main__':
    app.run(debug=True)
