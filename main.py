import joblib
import numpy as np
pipe_lr = joblib.load(open("emotion_detection.pkl", "rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    raw_text = input("Enter the text: ")

    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)

    print(f"Original Text: {raw_text}")
    print(f"Prediction: {prediction}")
    print(f"Prediction Probability: {probability}")

if __name__ == '__main__':
    main()
