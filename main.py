# -*- coding: utf-8 -*-
"""
Prilikom upotrebe softvera obavezno je citiranje radova navedenih uz njega i stranice ReLDI repozitorijuma
"""
import classla

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from funcs import *



file_name = "tekst.txt"
X = tfidf(file_name)

Um, Sm, Vt = np.linalg.svd(X)
print(Um.shape,"\n")
print(Sm.shape,"\n")
print(Vt.shape)



#import matplotlib.pyplot as plt
#
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True

(Um, Sm, Vt) = reduce(Um, Sm, Vt, 0.975)


#x = list(range(0, len(adj_sim)))
#plt.title("Adjacent sentence graph")
#plt.plot(x, adj_sim, color="black")
#plt.plot(x, adj_sim_norm, color="blue")
#plt.show()
#plt.title("depth graph")
#plt.plot(x, depth_norm, color="black")
#plt.show()

ss = SS(Vt, Sm)
adj_sim = calculate_adjacent_similarity(ss)



'''
from classla import Pipeline
classla.download('sr')
#classla.download('sr', type='nonstandard')
nlp = Pipeline(lang='sr', processors='tokenize,pos,lemma')

sentence = "Ovo je primer za lematizaciju."
doc = nlp(' '.join(get_sentences_from_file("tekst.txt")))
#doc = nlp(sentence)
#print(doc)
'''