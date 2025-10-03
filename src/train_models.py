from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
from sklearn.naive_bayes import MultinomialNB
from preprocess import readDataSet
import matplotlib.pyplot as plt
import seaborn as sns


X = readDataSet['cleaned']
y = readDataSet[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


def evaluate_model(algorithm, algorithm_name):
    model = Pipeline([('tfidf', TfidfVectorizer(max_features=3000)), ('clf', algorithm)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {algorithm_name} Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='spam'))
    print("Recall:", recall_score(y_test, y_pred, pos_label='spam'))
    print("F1-Score:", f1_score(y_test, y_pred, pos_label='spam'))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))


    return model, y_pred

def visualize_confusion_matrix(model,algorithm_name):
    y_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    cf_matrix_div = cf_matrix.astype('float') / cf_matrix.sum()
    plt.figure(figsize=(10, 7))
    sns.heatmap(cf_matrix_div, annot=True,
     fmt='.2%', cmap='Blues')
    plt.title(f"{algorithm_name} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def save_model(model,algorithm_name):
   import os

   os.makedirs('models', exist_ok=True)
   
   filename = f"models/{algorithm_name.lower().replace(' ','-')}_model.pkl"
   with open(filename, 'wb') as f:
       pickle.dump(model,f)
   print(f"Model saved to {filename}")
   return filename



model1,y_pred1 = evaluate_model(LogisticRegression(max_iter=1000, solver='liblinear'), "Logistic Regression")
save_model(model1,"Logistic Regression")
visualize_confusion_matrix(model1,"Logistic Regression")
model2,y_pred2 = evaluate_model(MultinomialNB(), "Naive Bayes")
save_model(model2,"Naive Bayes")
visualize_confusion_matrix(model2,"Naive Bayes")
print("Comparing Logistic Regression and Naive Bayes Models")
acc1=accuracy_score(y_test, y_pred1)
acc2=accuracy_score(y_test, y_pred2)

if(acc2>acc1):
    print("Naive Bayes is better")
else:
    print("Logistic Regression is better")
if recall_score(y_test, y_pred2, pos_label='spam') > recall_score(y_test, y_pred1, pos_label='spam'):
      print("Naive Bayes catches more spam")
