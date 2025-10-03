from nltk import word_tokenize
import preprocess
from collections import Counter


def mostFrequentWords(dataSet,n=5):
    all_words = []
    for word in dataSet:
        if ' ' in word:
         words = word.split()
        else:
            words = word_tokenize(word.lower())
        all_words.extend([w for w in  words if w.isalnum()])
    return  (Counter(all_words).most_common(n))

def mostFrequentWordsInSpam(dataSet,n=5):
    spamwords = dataSet[dataSet[0]=='spam']['cleaned']
    return (mostFrequentWords(spamwords,n))


def mostFrequentWordsInHam(dataSet,n=5):
    hamwords = dataSet[dataSet[0]=='ham']['cleaned']
    return (mostFrequentWords(hamwords,n))



print("The most frequent words in the dataset")
print(mostFrequentWords(preprocess.readDataSet['cleaned']))
print("\nThe most frequent words in spam")
print(mostFrequentWordsInSpam(preprocess.readDataSet))
print("\nThe most frequent words in ham")
print(mostFrequentWordsInHam(preprocess.readDataSet))