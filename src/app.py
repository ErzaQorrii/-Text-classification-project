import os
import pickle
from flask  import Flask,request,jsonify
from .preprocess import clean_text

with open("models/naive-bayes_model.pkl","rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <html>
        <body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
            <h1>SMS Spam Detector</h1>
            <form method="POST" action="/predict_form">
                <textarea name="message" rows="4" cols="50" placeholder="Enter your SMS message here..." required></textarea>
                <br><br>
                <button type="submit" style="padding: 10px 20px; font-size: 16px;">Check if Spam</button>
            </form>
        </body>
    </html>
    '''
    
@app.route('/predict_form', methods=['POST'])
def predict_form():
    message = request.form.get('message', '')
    cleaned = clean_text(message)
    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0]
    spam_prob = probability[1] * 100
    
    result_html = f'''
    <html>
        <body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
            <h1>Result</h1>
            <p><strong>Message:</strong> {message}</p>
            <p><strong>Prediction:</strong> <span style="color: {'red' if prediction == 'spam' else 'green'}; font-size: 24px;">{prediction.upper()}</span></p>
            <p><strong>Spam Probability:</strong> {spam_prob:.1f}%</p>
            <br>
            <a href="/">Check another message</a>
        </body>
    </html>
    '''
    return result_html

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data.get('message','')
    cleaned = clean_text(message)
    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0]
    return jsonify({
        'message': message,
        'prediction': prediction,
        'spam_probability': float(probability[1]),
        'is_spam': prediction == 'spam'
     })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=False)



