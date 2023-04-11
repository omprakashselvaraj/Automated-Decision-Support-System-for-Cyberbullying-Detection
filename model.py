# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:42:30 2023

@author: omprakash
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
import pickle
import pandas as pd

class Model:
    def __init__(self, df):
        self.data = df
        self.x = self.data['tweet_text']
        self.y = self.data['cyberbullying_type']
        
    def train_test_splitter(self):
        x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.3,random_state=101)
        return x_train,x_test,y_train,y_test
    
    def tfidf_vectorization(self,x_train,x_test,y_train,y_test):
        tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
        tfidf_test=tfidf_vectorizer.transform(x_test)
        Pkl_filename = "tfidf.pkl"  
        pickle.dump(tfidf_vectorizer, open(Pkl_filename, 'wb'))
        return tfidf_train, tfidf_test
    
    def count_vectorization(self,x_train,x_test,y_train,y_test):
        count_vectorizer=CountVectorizer(stop_words='english', max_df=0.7)
        count_train=count_vectorizer.fit_transform(x_train) 
        count_test=count_vectorizer.transform(x_test)
        Pkl_filename = "count.pkl"  
        pickle.dump(count_vectorizer, open(Pkl_filename, 'wb'))
        return count_train, count_test
    
    def logistic_regression(self,vector_train, vector_test, y_train, y_test):
        lr=LogisticRegression(random_state=0)
        lr.fit(vector_train,y_train)
        lr_train=lr.predict(vector_train)
        lr_test = lr.predict(vector_test)
        return lr_train, lr_test
    
    def naivebayes_classifier(self,vector_train, vector_test, y_train, y_test):
        nb = naive_bayes.MultinomialNB()
        nb.fit(vector_train,y_train)
        nb_train=nb.predict(vector_train)
        nb_test = nb.predict(vector_test)
        return nb_train, nb_test
    
    def evaluation_metrics(self,lr_train, lr_test,x_train,x_test,y_train,y_test,model):
        acc_tr = accuracy_score(lr_train,y_train)
        f1_tr = f1_score(lr_train,y_train,average = 'weighted')
        precision_tr = precision_score(lr_train,y_train, average = 'weighted')
        recall_tr = recall_score(lr_train,y_train, average = 'weighted')
        
        acc_te = accuracy_score(lr_test,y_test)
        f1_te = f1_score(lr_test,y_test,average = 'weighted')
        precision_te = precision_score(lr_test,y_test, average = 'weighted')
        recall_te = recall_score(lr_test,y_test, average = 'weighted')
        
        pm = {
            'model' : [model,model],
            'data' : ['Trainig data','Testing data'],
            'accuracy':[acc_tr,acc_te],
            'precision':[precision_tr,precision_te],
            'recall':[recall_tr,recall_te],
            'f1 score':[f1_tr,f1_te]
                
            }
        res = pd.DataFrame.from_dict(pm)
        return res
        
        
    
    

