# -*- coding: utf-8 -*-
"""
Prilikom upotrebe softvera obavezno je citiranje radova navedenih uz njega i stranice ReLDI repozitorijuma
"""
from classla import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from funcs import *
import random



(sentences , sums) = get_sentances_and_sums_from_json("dataset.json")
nlp = Pipeline(lang='sr', processors='tokenize,pos,lemma')
con_indexes = []
for i in range(1):
    rand_indexes = random.sample(range(50), 5)
    sentences_concat = []
    for r in rand_indexes:
        sentences_concat += sentences[r]
    print(sentences_concat)
    X = tfidf(lematize_list_of_sentences(sentences_concat, nlp))
    Um, Sm, Vt = np.linalg.svd(X)
    #print(Um.shape,"\n")
    #print(Sm.shape,"\n")
    #print(Vt.shape)

    (Um, Sm, Vt) = reduce(Um, Sm, Vt, 0.975)
    ss = SS(Vt, Sm)
    extracted = extract_sentances(ss, 0.4, depth_improved_function, 0.4)
    print(len(extracted), " ", len(sentences_concat))




#classla.download('sr')
#classla.download('sr', type='nonstandard')

