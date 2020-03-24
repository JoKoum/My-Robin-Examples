import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import os.path

def ModelTrain():
    df = pd.read_csv('moviereviews2.tsv',sep='\t')

    blanks = []

    for ind,label,review in df.itertuples():
        if type(review) == str:
            if review.isspace():
                blanks.append(ind)


    df.dropna(inplace=True)

    df['label'].value_counts()

    X = df['review']
    y = df['label']


    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

    text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
    text_clf.fit(X_train,y_train)
    joblib.dump(text_clf,'sentiment_model.sav')
    
def SentimentAnalysis(text):
    
    text_clf = joblib.load('sentiment_model.sav')
    prediction = text_clf.predict(text)
    
    return prediction
    
if __name__ == "__main__":
    
    if os.path.exists('sentiment_model.sav'):
        print(SentimentAnalysis([sys.argv[1]]))
    else:
        ModelTrain()
        print(SentimentAnalysis([sys.argv[1]]))
