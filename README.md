# Handmade_Naive_Bayes

Code is handwritten version of Naive Bayes Algorithm in python. 
Main focus here is obtaining naive bayes algorithm without using ani library in python.
Test and Train datas are not included here but can easily obtain from (https://www.kaggle.com/c/fake-news/data?select=submit.csv).
This code is Multinomial approach of Naive Bayes, designed for text classification.
Program explained in powerpoint presentation with its fundamentals.
For big datasets program works really slower (in my case 35-40 minutes). To prevent waiting I saved unique word dictionary as "unique_words_freqs" using pickle library in python. Also test set's output values were not given properly from kaggle that's why I create "testy" file too.
