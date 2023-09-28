# -*- coding: utf-8 -*-
"""
Prilikom upotrebe softvera obavezno je citiranje radova navedenih uz njega i stranice ReLDI repozitorijuma
"""
import classla

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from funcs import *



sentences = ["recenica", "another one"]
X = tfidf(sentences)

Um, Sm, Vt = np.linalg.svd(X)
print(Um.shape,"\n")
print(Sm.shape,"\n")
print(Vt.shape)

(Um, Sm, Vt) = reduce(Um, Sm, Vt, 0.975)
ss = SS(Vt, Sm)

extracted = extract_sentances(ss, 0.4, depth_improved_function, 0.4)
for i in extracted:
    print(sentences[i])
'''
from classla import Pipeline
classla.download('sr')
#classla.download('sr', type='nonstandard')
nlp = Pipeline(lang='sr', processors='tokenize,pos,lemma')

sentence = "Ovo je primer za lematizaciju."
doc = nlp(' '.join(get_sentences_from_file("tekst.txt")))
lemma = doc.sentences[0].words[0].lemma
lemmas = [word.lemma for word in doc.sentences[0].words]
#doc = nlp(sentence)
#print(doc)
'''