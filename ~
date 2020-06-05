import numpy
import pandas as pd


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
from pdb import set_trace as bp


#df =pd.read_csv('universalnames_dataset.csv')
df =pd.read_csv('names_dataset.csv')
print(df.head())
print(df.size)
print(df.columns)
print(df.dtypes)

print(df.isnull().isnull().sum())

print(df[df.sex == 'F'].size)

print(df[df.sex == 'M'].size)

df_names = df
#df_names = df_names.dropna()
df_names.sex.replace({'F':0, 'M':1}, inplace=True)
#df_names.sex.replace({'NaN':0}, inplace=True)
df_names.sex.unique()

Xfeatures =df_names['name']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
print(X)
#print(cv.get_feature_names())
bp()
y = df_names.sex

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")
print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")

### sample prediction

sample_name = ["Mary"]
vect = cv.transform(sample_name).toarray()
clf.predict(vect)

def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

genderpredictor("Martha")        

NaiveBayesModel = open("naivebayesgendermodel.pkl","wb")
joblib.dump(clf,NaiveBayesModel)
NaiveBayesModel.close()

count_vec = open("Count_vector.pkl", "wb")
joblib.dump(cv,count_vec)
count_vec.close()
