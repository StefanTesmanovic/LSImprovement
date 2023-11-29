from classla import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from funcs import *
import random
import json

(sentences , sums) = get_sentances_and_sums_from_json("dataset.json")
nlp = Pipeline(lang='sr', processors='tokenize,pos,lemma')
for i in range(5, 46, 5):
    print(i)
    num_of_texts = i
    avg_tfidf_o = 0
    avg_tfidf_lem_o = 0
    avg_countvec_lem_o = 0
    avg_countvec_o = 0
    avg_tfidf_n = 0
    avg_tfidf_lem_n = 0
    avg_countvec_lem_n = 0
    avg_countvec_n = 0
    for j in range(50):
        if j % 5 == 0:
            print(j)
        rand_indexes = random.sample(range(50), num_of_texts)
        sentences_concat = []
        for r in rand_indexes:
            sentences_concat += sentences[r]
        X_tfidf_lem = tfidf(lematize_list_of_sentences(sentences_concat, nlp))
        X_countvec_lem = countvec(lematize_list_of_sentences(sentences_concat, nlp))
        X_countvec = countvec(sentences_concat)
        X_tfidf = tfidf(sentences_concat)
        
        avg_tfidf_o += average_alfa(X_tfidf, depth_old, num_of_texts)
        avg_tfidf_lem_o += average_alfa(X_tfidf_lem, depth_old, num_of_texts)
        avg_countvec_lem_o += average_alfa(X_countvec_lem, depth_old, num_of_texts)
        avg_countvec_o += average_alfa(X_countvec, depth_old, num_of_texts)
        avg_tfidf_n += average_alfa(X_tfidf, depth_improved_function, num_of_texts)
        avg_tfidf_lem_n += average_alfa(X_tfidf_lem, depth_improved_function, num_of_texts)
        avg_countvec_lem_n += average_alfa(X_countvec_lem, depth_improved_function, num_of_texts)
        avg_countvec_n += average_alfa(X_countvec, depth_improved_function, num_of_texts)

avg_tfidf_o /= 9*50
avg_tfidf_lem_o /= 9*50
avg_countvec_lem_o /= 9*50
avg_countvec_o /= 9*50
avg_tfidf_n /= 9*50
avg_tfidf_lem_n /= 9*50
avg_countvec_lem_n /= 9*50
avg_countvec_n /= 9*50

print(avg_tfidf_o)
print(avg_tfidf_lem_o)
print(avg_countvec_lem_o)
print(avg_countvec_o)
print(avg_tfidf_n)
print(avg_tfidf_lem_n)
print(avg_countvec_lem_n)
print(avg_countvec_n)

# Create a dictionary with variable names as keys and their values
data = {
    "avg_tfidf_o": avg_tfidf_o,
    "avg_tfidf_lem_o": avg_tfidf_lem_o,
    "avg_countvec_lem_o": avg_countvec_lem_o,
    "avg_countvec_o": avg_countvec_o,
    "avg_tfidf_n": avg_tfidf_n,
    "avg_tfidf_lem_n": avg_tfidf_lem_n,
    "avg_countvec_lem_n": avg_countvec_lem_n,
    "avg_countvec_n": avg_countvec_n
}

# Specify the file path where you want to save the JSON data
file_path = "avg_alfa_values.json"

# Write the data to the JSON file
with open(file_path, 'w') as json_file:
    json.dump(data, json_file, indent=2)

print(f"Values written to {file_path}")


