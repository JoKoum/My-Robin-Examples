import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

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

def SentimentAnalysis(text):

    prediction = text_clf.predict(text)
    
    return prediction
    
if __name__ == "__main__":
    print(SentimentAnalysis([sys.argv[1]]))
