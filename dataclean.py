import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from num2words import num2words
from bs4 import BeautifulSoup
import re
import unidecode
from string import punctuation
import nltk
from emoticons_list import EMOTICONS_EMO
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import tokenize

class Dataclean:
    def __init__(self,file):
        self.data = pd.read_csv(file)
        
    def data_properties(self):
        print("Shape of Dataset ", self.data.shape)
        print("Attributes in Dataset ", self.data.columns)
        print("Unique Class in Cyber Bullying ",self.data['cyberbullying_type'].unique())
        print("Count for each class in cyberbullying type ")
        print(self.data['cyberbullying_type'].value_counts())
        
    def count_plot(self,column):
        ax = sns.countplot(x = column , data = self.data)
        fig = ax.get_figure()
        return fig
    
    def lower_case_convertion(self,text):
        lower_text = text.lower()
        return lower_text
    
    def remove_punctuation(self,text):
        """
        Return :- String after removing punctuations
        Input :- String
        Output :- String
        """
        return text.translate(str.maketrans('', '', punctuation))

    def numtowords(self,text):
        """
        Return :- text which have all numbers or integers in the form of words
        Input :- string
        Output :- string
        """
        # splitting text into words with space
        after_spliting = text.split()

        for index in range(len(after_spliting)):
            if after_spliting[index].isdigit():
                after_spliting[index] = num2words(after_spliting[index])

        # joining list into string with space
        numbers_to_words = ' '.join(after_spliting)
        return numbers_to_words
    
    def remove_html_tags_beautifulsoup(self,text):
        """
        Return :- String without Html tags
        input :- String
        Output :- String
        """
        parser = BeautifulSoup(text, "html.parser")
        without_html = parser.get_text(separator = " ")
        return without_html
    
    def remove_urls(self,text):
        """
        Return :- String without URLs
        input :- String
        Output :- String
        """
        url_pattern = r'https?://\S+|www\.\S+'
        without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
        return without_urls
    
    def accented_to_ascii(self,text):
        """
        Return :- text after converting accented characters
        Input :- string
        Output :- string
        """
        # apply unidecode function on text to convert
        # accented characters to ASCII values
        text = unidecode.unidecode(text)
        return text
    
    def remove_extra_spaces(self,text):
        """
        Return :- string after removing extra whitespaces
        Input :- String
        Output :- String
        """
        space_pattern = r'\s+'
        without_space = re.sub(pattern=space_pattern, repl=" ", string=text)
        return without_space
    
    def remove_single_char(self,text):
        """
        Return :- string after removing single characters
        Input :- string
        Output:- string
        """
        single_char_pattern = r'\s+[a-zA-Z]\s+'
        without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
        return without_sc
    
    
    def emoji_words(self,text):
        UNICODE_EMO = {v: k for k, v in EMOTICONS_EMO.items()}
        for emot in UNICODE_EMO:
            emoji_pattern = r'('+emot+')'
            # replace
            emoji_words = UNICODE_EMO[emot]
            replace_text = emoji_words.replace(",","")
            replace_text = replace_text.replace(":","")
            replace_text_list = replace_text.split()
            emoji_name = '_'.join(replace_text_list)
            text = re.sub(emoji_pattern, emoji_name, text)
        return text
    
    def lemmatization(self,text):
        lemma = WordNetLemmatizer()
        # word tokenization
        tokens = word_tokenize(text)

        for index in range(len(tokens)):
            # lemma word
            lemma_word = lemma.lemmatize(tokens[index])
            tokens[index] = lemma_word

        return ' '.join(tokens)
    
    token_space = tokenize.WhitespaceTokenizer()

    def counter(self, text, column_text, quantity):
        token_space = tokenize.WhitespaceTokenizer()
        all_words = ' '.join([text for text in text[column_text]])
        token_phrase = token_space.tokenize(all_words)
        frequency = nltk.FreqDist(token_phrase)
        df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                       "Frequency": list(frequency.values())})
        df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
        plt.figure(figsize=(12,8))
        ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'cyan')
        ax.set(ylabel = "Count")
        plt.xticks(rotation='vertical')
        plt.show()
        
    def label(self,x):
        label={'religion': 0,     
        'age': 1,                    
        'gender':2,                 
        'ethnicity':3,              
        'not_cyberbullying':4,      
        'other_cyberbullying':5}
        return label[x]
            