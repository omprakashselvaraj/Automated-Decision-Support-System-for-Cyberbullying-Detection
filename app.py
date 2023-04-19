from flask import Flask, render_template, request, redirect, url_for
from nltk.corpus import stopwords
import dataclean
import pickle
from datetime import datetime

app = Flask(__name__)
file = 'cyberbullying_tweets.csv'
dc = dataclean.Dataclean(file)

path='lg.pkl'
model = pickle.load(open(path, 'rb'))
path1='count.pkl'
vector=pickle.load(open(path1,'rb'))

@app.route('/')
def mainpage():
    return render_template('form.html')    

@app.route('/input', methods = ['GET','POST'])
def get_input():
    details = request.form
    input_text = details['text']
    return redirect(url_for('pre_process', x = input_text))

@app.route('/process/<x>')
def pre_process(x):
    start = datetime.now()
    x = dc.lower_case_convertion(x)
    x = dc.remove_punctuation(x)
    x = dc.numtowords(x)
    x = dc.lower_case_convertion(x)
    x = dc.remove_html_tags_beautifulsoup(x)
    x = dc.remove_urls(x)
    x = dc.accented_to_ascii(x)
    x = dc.remove_extra_spaces(x)
    x = dc.remove_single_char(x)
    stop = stopwords.words('english')
    x= ' '.join([word for word in x.split() if word not in (stop)])
    x = dc.emoji_words(x)
    x = dc.lemmatization(x)
    x = [x]
    vect = vector.transform(x).toarray()
    my_prediction = model.predict(vect)
    val = my_prediction[0]
    label={0:'religion',     
    1:'age',                    
    2:'gender',                 
    3:'ethnicity',              
    4:'not_cyberbullying',      
    5:'other_cyberbullying'}
    msg = label[val]
    end = datetime.now()
    td = (end - start).total_seconds() 
    print(f"The time of execution of above program is : {td:.03f}ms")
    return render_template('output.html', msg = msg)
    


if __name__ == '__main__':
    app.run(debug=())