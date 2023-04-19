import dataclean
import model
from nltk.corpus import stopwords
import pandas as pd

if __name__ == '__main__':
    file = 'cyberbullying_tweets.csv'
    vector = 'count'
    '''models = ['logistic_regression','naivebayes_classifier',
              'decisiontree_classifier','randomforest_classifier',
              'svm_classifier','passiveaggressive_classifier']'''
    models = ['logistic_regression']
    dc = dataclean.Dataclean(file)
    dc.data_properties()
    '''plot = dc.count_plot('cyberbullying_type')'''
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.lower_case_convertion)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.remove_punctuation)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.numtowords)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.lower_case_convertion)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.remove_html_tags_beautifulsoup)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.remove_urls)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.accented_to_ascii)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.remove_extra_spaces)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.remove_single_char)
    stop = stopwords.words('english')
    dc.data['tweet_text'] =dc. data['tweet_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.emoji_words)
    dc.data['tweet_text'] = dc.data['tweet_text'].apply(dc.lemmatization)
    print(dc.data.head())
    
    '''
    dc.counter(dc.data[dc.data["cyberbullying_type"] == "not_cyberbullying"], "tweet_text", 40)
    dc.counter(dc.data[dc.data["cyberbullying_type"] == "gender"], "tweet_text", 40)
    dc.counter(dc.data[dc.data["cyberbullying_type"] == "age"], "tweet_text", 40)
    dc.counter(dc.data[dc.data["cyberbullying_type"] == "religion"], "tweet_text", 40)
    dc.counter(dc.data[dc.data["cyberbullying_type"] == "ethicity"], "tweet_text", 40)
    dc.counter(dc.data[dc.data["cyberbullying_type"] == "other_cyberbullying"], "tweet_text", 40)
    '''
    
    dc.data['cyberbullying_type'] = dc.data['cyberbullying_type'].apply(dc.label)
    print(dc.data.head())
    
    data = dc.data
    mod = model.Model(data)
    model_dic = {'logistic_regression':mod.logistic_regression,
                 'naivebayes_classifier':mod.naivebayes_classifier,
                 'decisiontree_classifier':mod.decisiontree_classifier,
                 'randomforest_classifier':mod.randomforest_classifier,
                 'svm_classifier':mod.svm_classifier,
                 'passiveaggressive_classifier':mod.passiveAggressive_classifier,
                 'xgboost_classifier': mod.xgboost_classifier,
                 'lgbm_classifier':mod.lgbm_classifier
                 }
    x_train,x_test,y_train,y_test = mod.train_test_splitter()
    if vector == 'tfidf':
        vector_train, vector_test = mod.tfidf_vectorization(x_train, x_test, y_train, y_test)
    else:
        vector_train, vector_test = mod.count_vectorization(x_train, x_test, y_train, y_test)
    
    '''pre_train, pre_test = mod.logistic_regression(vector_train, vector_test, y_train, y_test)
    res = mod.evaluation_metrics(pre_train, pre_test, x_train, x_test, y_train, y_test)
    print(res)'''
    result = pd.DataFrame()
    for i in models:
        pre_train, pre_test = model_dic[i](vector_train, vector_test, y_train, y_test)
        res = mod.evaluation_metrics(pre_train, pre_test, x_train, x_test, y_train, y_test,i)
        result = result.append(res, ignore_index = True)
    print(result)
    result.to_excel('result3.xlsx',index = False)
        
        
