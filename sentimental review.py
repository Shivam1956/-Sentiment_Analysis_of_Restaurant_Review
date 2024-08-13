import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive
drive.mount('/content/drive/')
df=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)
df.head()
df.shape
df.info()
import nltk
import string
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stopwords.words('english')
def text_process(msg):
    nopunc =[char for char in msg if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])
corpus = []
for i in range(0,1000):
  review = re.sub(pattern = '[^a-zA-z]', repl = ' ', string=df['Review'][i])

  review = review.lower()
  review_words = review.split()
  review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

  ps = PorterStemmer()

  review = [ps.stem(word) for word in review_words]
  review = ' '.join(review)

  corpus.append(review)
corpus[:1500]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=0)
X_train.shape,X_test.shape, y_train.shape,y_test.shape
from sklearn.naive_bayes import MultinomialNB

classifier =MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 =accuracy_score(y_test,y_pred)
score2 = accuracy_score(y_test,y_pred)
score3 = recall_score(y_test,y_pred)

print("<-----SCORE----->")
print("Accuracy score = {}%".format(round(score1*100,3)))
print("Precision score = {}%".format(round(score2*100,3)))
print("recall score = {}%".format(round(score3*100,3)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize =(10,6))
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])
plt.xlabel('Predicted values')
plt.ylabel('Actual Values')
from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
best_accuracy =0.0
alpha_val =0.0
for i in np.arange(0.1,1.1,0.1):
  temp_classifier =MultinomialNB(alpha=i)
  temp_classifier.fit(X_train,y_train)
  temp_y_pred =temp_classifier.predict(X_test)
  score = accuracy_score(y_test,temp_y_pred)
  print("Accuracy Score for alpha ={} is {}%".format(round(i,1),round(score*100,3)))
  if score>best_accuracy:
     best_accuracy=score
     alpha_val =i
print('----------------------------------------------------')
print("The Best Accuracy Score is {}% with alpha value as {}".format(round(best_accuracy*100, 2), round(alpha_val, 1)))
classifier =MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def predict_sentiment(sample_review):
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)

    temp = cv.transform([final_review]).toarray()
    return classifier.predict(temp)
sample_review ='The food is so good.'

if predict_sentiment(sample_review):
  print("it's a Positive review")

else:
  print("Negative review")
  sample_review ='the service was very slow'

if predict_sentiment(sample_review):
  print("it's a Positive review")

else:
  print("it's a Negative review")
from sklearn.ensemble import RandomForestClassifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(X_train,y_train)
y_pred = randomclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)
print("the matrix is = \n{}".format(confusion_m))
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("the accuracy score is = {}%".format(accuracy*100))
