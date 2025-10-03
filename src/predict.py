import pickle
from preprocess import clean_text


def load_model():
  with open('models/naive-bayes_model.pkl','rb') as f :
      model = pickle.load(f)
  return model

def getStopwords():
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

model = load_model()
stopwords = getStopwords()

def predict(message):
  cleaned_message = clean_text(message)
  print(f"Original: {message}")
  print(f"Cleaned: {cleaned_message}")
  prediction = model.predict([cleaned_message])[0]
  proba = model.predict_proba([cleaned_message])[0]
  print(f"Spam probability: {proba[1]:.2%}")
  return prediction

def main():
    message = input("Enter your message: ")
    result = predict(message)
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()