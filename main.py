# -*- coding: utf-8 -*-
"""
Prilikom upotrebe softvera obavezno je citiranje radova navedenih uz njega i stranice ReLDI repozitorijuma
"""
from classla import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from funcs import *
import random
from rouge import Rouge

rouge = Rouge()

iterations = 200
num_of_texts = 40

(sentences , sums) = get_sentances_and_sums_from_json("dataset.json")
nlp = Pipeline(lang='sr', processors='tokenize,pos,lemma')
con_indexes = []
rouge_1_scores = []
rouge_2_scores = []
rouge_l_scores = []
rand_index_list = []
average_rouge_1 = np.zeros(8)
average_rouge_1_p = np.zeros(8)
average_rouge_1_r = np.zeros(8)
average_rouge_2 = np.zeros(8)
average_rouge_2_p = np.zeros(8)
average_rouge_2_r = np.zeros(8)
average_rouge_l = np.zeros(8)
average_rouge_l_p = np.zeros(8)
average_rouge_l_r = np.zeros(8)
old_par_split_treshold = 1
new_par_split_treshold = 0.4
for i in range(iterations):
    if i % 5 == 0:
        print(i,"*********************************")
    rand_indexes = random.sample(range(50), num_of_texts)
    rand_index_list.append(rand_indexes)
    sentences_concat = []
    sums_concat = []
    for r in rand_indexes:
        sentences_concat += sentences[r]
        sums_concat += sums[r]

    rouge_l_scores_temp = []
    rouge_1_scores_temp = []
    rouge_2_scores_temp = []

    X_tfidf_lem = tfidf(lematize_list_of_sentences(sentences_concat, nlp))
    X_countvec_lem = countvec(lematize_list_of_sentences(sentences_concat, nlp))
    X_countvec = countvec(sentences_concat)
    X_tfidf = tfidf(sentences_concat)

    gen_sums = []

    extracted = extract_sentances(X_countvec, 0.4, depth_old, old_par_split_treshold)
    sentences_extracted = [sentences_concat[j] for j in extracted]
    gen_sums.append(sentences_extracted)

    extracted_improved = extract_sentances(X_countvec, 0.4, depth_improved_function, new_par_split_treshold)
    sentences_extracted_improved = [sentences_concat[j] for j in extracted_improved]
    gen_sums.append(sentences_extracted_improved)

    extracted_lem = extract_sentances(X_countvec_lem, 0.4, depth_old, old_par_split_treshold)
    sentences_extracted_lem = [sentences_concat[j] for j in extracted_lem]
    gen_sums.append(sentences_extracted_lem)

    extracted_improved_lem = extract_sentances(X_countvec_lem, 0.4, depth_improved_function, new_par_split_treshold)
    sentences_extracted_improved_lem = [sentences_concat[j] for j in extracted_improved_lem]
    gen_sums.append(sentences_extracted_improved_lem)

    extracted_tfidf = extract_sentances(X_tfidf, 0.4, depth_old, old_par_split_treshold)
    sentences_extracted_tfidf = [sentences_concat[j] for j in extracted_tfidf]
    gen_sums.append(sentences_extracted_tfidf)

    extracted_tfidf_improved = extract_sentances(X_tfidf, 0.4, depth_improved_function,new_par_split_treshold)
    sentences_extracted_tfidf_improved = [sentences_concat[j] for j in extracted_tfidf_improved]
    gen_sums.append(sentences_extracted_tfidf_improved)

    extracted_tfidf_lem = extract_sentances(X_tfidf_lem, 0.4, depth_old, old_par_split_treshold)
    sentences_extracted_tfidf_lem = [sentences_concat[j] for j in extracted_tfidf_lem]
    gen_sums.append(sentences_extracted_tfidf_lem)

    extracted_tfidf_improved_lem = extract_sentances(X_tfidf_lem, 0.4, depth_improved_function, new_par_split_treshold)
    sentences_extracted_tfidf_improved_lem = [sentences_concat[j] for j in extracted_tfidf_improved_lem]
    gen_sums.append(sentences_extracted_tfidf_improved_lem)
    
    j = 0
    for sum in gen_sums:
        scores = rouge.get_scores(" ".join(sum), " ".join(sums_concat), avg=False)
        if i %10 == 0:
            print(scores)
        rouge_1_scores_temp.append(scores[0]['rouge-1']['f'])
        average_rouge_1[j] += scores[0]['rouge-1']['f']
        average_rouge_1_p[j] += scores[0]['rouge-1']['p']
        average_rouge_1_r[j] += scores[0]['rouge-1']['r']

        rouge_2_scores_temp.append(scores[0]['rouge-2']['f'])
        average_rouge_2[j] += scores[0]['rouge-2']['f']
        average_rouge_2_p[j] += scores[0]['rouge-2']['p']
        average_rouge_2_r[j] += scores[0]['rouge-2']['r']

        rouge_l_scores_temp.append(scores[0]['rouge-l']['f'])
        average_rouge_l[j] += scores[0]['rouge-l']['f']
        average_rouge_l_p[j] += scores[0]['rouge-l']['p']
        average_rouge_l_r[j] += scores[0]['rouge-l']['r']
        j += 1
    if i % 10 == 0:
        print(average_rouge_1 /   (i+1))
        print(average_rouge_1_p / (i+1))
        print(average_rouge_1_r / (i+1))
        print(average_rouge_2 /   (i+1))
        print(average_rouge_2_p / (i+1))
        print(average_rouge_2_r / (i+1))
        print(average_rouge_l /   (i+1))
        print(average_rouge_l_p / (i+1))
        print(average_rouge_l_r / (i+1))
    rouge_1_scores.append(rouge_1_scores_temp)
    rouge_2_scores.append(rouge_2_scores_temp)
    rouge_l_scores.append(rouge_l_scores_temp)


file_path = "results_40texts_200_alf1-04.txt"

rand_index_list = np.array(rand_index_list)
rouge_1_scores = np.array(rouge_1_scores)
rouge_2_scores = np.array(rouge_2_scores)
rouge_l_scores = np.array(rouge_l_scores)

average_rouge_1 /= iterations
average_rouge_1_p /= iterations
average_rouge_1_r /= iterations
average_rouge_2 /= iterations
average_rouge_2_p /= iterations
average_rouge_2_r /= iterations
average_rouge_l /= iterations
average_rouge_l_p /= iterations
average_rouge_l_r /= iterations

with open(file_path, 'w') as file:
    file.write("indexes of concatenated sums:\n")
    file.write(np.array2string(rand_index_list, separator=', ') + '\n\n')

    file.write("rouge 1:\n")
    file.write(np.array2string(rouge_1_scores, separator=', ') + '\n\n')
    
    file.write("rouge 2:\n")
    file.write(np.array2string(rouge_2_scores, separator=', ') + '\n\n')
    
    file.write("rouge l:\n")
    file.write(np.array2string(rouge_l_scores, separator=', ') + '\n\n')

    file.write("averages f score 1, 2, l:\n")
    file.write(np.array2string(average_rouge_1, separator=', ') + '\n')
    file.write(np.array2string(average_rouge_2, separator=', ') + '\n')
    file.write(np.array2string(average_rouge_l, separator=', ') + '\n\n')

    file.write("averages precision 1, 2, l:\n")
    file.write(np.array2string(average_rouge_1_p, separator=', ') + '\n')
    file.write(np.array2string(average_rouge_2_p, separator=', ') + '\n')
    file.write(np.array2string(average_rouge_l_p, separator=', ') + '\n\n')

    file.write("averages recall 1, 2, l:\n")
    file.write(np.array2string(average_rouge_1_r, separator=', ') + '\n')
    file.write(np.array2string(average_rouge_2_r, separator=', ') + '\n')
    file.write(np.array2string(average_rouge_l_r, separator=', ') + '\n')


print("rouge scores documented to: ", file_path)



#classla.download('sr')
#classla.download('sr', type='nonstandard')

