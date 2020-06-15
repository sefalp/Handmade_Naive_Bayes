# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:11:58 2020

@author: alps
"""


import pandas as pd 
from nltk import tokenize
import re
import pickle


train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv") 


file = open("testy","rb")
testy = pickle.load(file)



file = open("unique_words_freqs","rb")
unique_words_freqs = pickle.load(file)
words_loaded = True


def organized_content(text):
    raw_content = str(text).lower()
    content = tokenize.sent_tokenize(raw_content)
    clean_punctuation = [re.sub(r'[^\w\s]', '', x) for x in content]
    organized_content = " ".join(clean_punctuation)
    last = organized_content.split()
    return last

def uniqueWordsDictionary(train):
    total_word_counter = 0                      #represents all word number in train data
    unique_words_freqs = {}                     #unique words and their frequencies for each label stored here
    for index in train.index:
        text = organized_content(train.loc[index]["text"])
        for word in text:
            total_word_counter += 1
            freq_array = [0,0]
            if(word not in unique_words_freqs.keys()):
                unique_words_freqs[word] = freq_array
            else:
                if(train.loc[index]["label"] == 0):
                    unique_words_freqs[word][0] += 1
                else:
                    unique_words_freqs[word][1] += 1
                

    word_count_fake = 0
    word_count_real = 0
    for index in train.index:
        text = organized_content(train.loc[index]["text"])
        if(train.loc[index]["label"] == 0):
            word_count_fake += len(text)
        else:
            word_count_real += len(text)


    
    unique_words_count = len(unique_words_freqs)
    for word in unique_words_freqs:
        prob_given_fake = ( unique_words_freqs[word][0] + 1 ) / (word_count_fake + unique_words_count)
        unique_words_freqs[word].append(prob_given_fake)    
        prob_given_real = ( unique_words_freqs[word][1] + 1 ) / (word_count_real + unique_words_count)
        unique_words_freqs[word].append(prob_given_real)

def MNB(test,testy,unique_words_freqs):
    count_zeros = 10387
    count_ones = 10413
    MNB_results = []
    for index in test.index:
        text = organized_content(test.loc[index]["text"])
        p_fake = count_zeros * (count_zeros + count_ones)
        p_real = count_ones * (count_zeros + count_ones)
        for word in text:
            if word in unique_words_freqs:
                p_fake = unique_words_freqs[word][2] * p_fake
                p_real = unique_words_freqs[word][3] * p_real
                if(p_fake < 10**(-10) or p_real < 10**(-10)):
                    p_fake *= 10**7
                    p_real *= 10**7
            else:
                p_fake *= 1
                p_real *= 1
        if(p_fake > p_real):
            MNB_results.append(0)
        else:
            MNB_results.append(1)
    
    testy["MNB_results"] = MNB_results
    
    success = 0
    fail = 0
    for index in testy.index:
        if(testy.loc[index]["labels"] == testy.loc[index]["MNB_results"]):
            success += 1
        else:
            fail += 1
    
    accuracy = success / (fail + success)
    return accuracy

def stop_words_removal():

    stop_words = [
    "a", "about", "above", "across", "after", "afterwards", 
    "again", "all", "almost", "alone", "along", "already", "also",    
    "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
    "this", "those", "though", "through", "throughout",
    "thru", "thus", "to", "together", "too", "toward", "towards",
    "under", "until", "up", "upon", "us",
    "very", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", 
    "who", "whoever", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
    ]
    
    for stop in stop_words:
        if(stop in unique_words_freqs):
            del unique_words_freqs[stop]


if(words_loaded == False):
    unique_words_freqs = uniqueWordsDictionary(train)

stop_words_removal()
accuracy = MNB(test, testy, unique_words_freqs)
print(accuracy)



        
        
        
        
        
    