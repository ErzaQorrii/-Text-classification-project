import string
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')


readDataSet = pd.read_csv("../dataset/sms+spam+collection/SMSSpamCollection", sep='\t', header=None)
print("\n========== Original Dataset Sample ==========")
print(readDataSet.head(5))
print("=" * 45)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    
    spam_keywords = {'won', 'win', 'winner', 'free', 'prize', 'call', 'claim', 'text', 'txt', 'urgent', 'offer'}
    stop_words = stop_words - spam_keywords
    
    text = text.lower()
    words = text.split()
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    return " ".join(filtered_words)



def clean_text(text):
     text = text.lower()

     for char in string.punctuation:
         text = text.replace(char, '')

     text = remove_stopwords(text)
     return text


readDataSet['cleaned']= readDataSet[1].apply(clean_text)
print('Data after cleaning')
print(readDataSet.head(5))

