# -*- coding: utf-8 -*-
"""
Prilikom upotrebe softvera obavezno je citiranje radova navedenih uz njega i stranice ReLDI repozitorijuma
"""
import classla

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def get_sentences_from_file(filename):
    sentences = []
    with open(filename, 'r') as file:
        text = file.read()
        pattern = r'\.{3}\n|\.{3}|\.\n|\. |[!] |[?] |[!]+\n|[?]+\n'
        sentences = re.split(pattern, text)
        sentences = [sentence.replace("\n", " ") for sentence in sentences]
        print(sentences)
        print(type(sentences))
    return sentences

def remove_zero_columns(X):
    # Find indices of columns with all zeros
    non_zero_columns = np.any(X != 0, axis=0)

    # Create a new matrix without columns of all zeros
    return X[:, non_zero_columns]

vectorizer = TfidfVectorizer()
file_name = "tekst.txt"
#X = vectorizer.fit_transform(corpus)
X = vectorizer.fit_transform(get_sentences_from_file(file_name))
X = X.toarray()
X = np.transpose(X)
X = remove_zero_columns(X)
#rint(X)
# = list(vectorizer.get_feature_names_out())
#rint(a)

Um, Sm, Vt = np.linalg.svd(X)
print(Um.shape,"\n")
print(Sm.shape,"\n")
print(Vt.shape)

def reduce(Um, Sm, Vt, l):
    sum = np.sum(Sm)
    for i in range(int(len(Sm)/2), len(Sm)):
      print(Sm[:i])
      if(i == len(Sm)-1):
          return(Um, Sm, Vt)
      if(np.sum(Sm[:i])/sum >= l):#ulazi ako prvih i vrednosti zadovoljava nejdnakost
        print(i)
        return (np.delete(Um,np.s_[i:],1), Sm[:i], np.delete(Vt, np.s_[i:], 0))

def WW(Um, Sm):
  return (Um * Sm**2) @ np.transpose(Um)

def SS(Vt, Sm):
    return (np.transpose(Vt) * Sm**2) @ Vt

def depth(ss):
    pass


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

(Um, Sm, Vt) = reduce(Um, Sm, Vt, 0.975)
ss = SS(Vt, Sm)
intensities = np.linalg.norm(ss, axis=0)
ss_norm = ss/intensities
adj_sim = np.diagonal(ss, offset=1)
adj_sim_norm = np.diagonal(ss_norm, offset=1)
x = list(range(0, len(adj_sim)))


plt.title("Adjacent sentence graph")
plt.plot(x, adj_sim, color="black")
plt.plot(x, adj_sim_norm, color="blue")
plt.show()

x = []
depth_norm = []
min = []
prev_l_max = adj_sim_norm[0]
prev_r_max = adj_sim_norm[len(adj_sim_norm)-1]
l_max = []
r_max = []
for i in range(1, len(adj_sim) - 1):
    if adj_sim_norm[i - 1] > adj_sim_norm[i] and adj_sim_norm[i] < adj_sim_norm[i + 1]:
        x.append(i)
        min.append(adj_sim_norm[i])
    if adj_sim_norm[i - 1] < adj_sim_norm[i] and adj_sim_norm[i] >  adj_sim_norm[i + 1]:
        l_max.append(adj_sim_norm[i])
        prev_l_max = adj_sim_norm[i]
    else:
        l_max.append(prev_l_max)
    if adj_sim_norm[len(adj_sim_norm)-i] < adj_sim_norm[len(adj_sim_norm)-1-i] and adj_sim_norm[len(adj_sim_norm)-1-i] >  adj_sim_norm[len(adj_sim_norm)-2-i]:
        r_max.append(adj_sim_norm[len(adj_sim_norm)-1-i])
        prev_r_max = adj_sim_norm[len(adj_sim_norm)-1-i]
    else:
        r_max.append(prev_r_max)
for i in range(0, len(x)):
    depth_norm.append((l_max[x[i]]+r_max[len(r_max)-1-x[i]])/(1-min[i]))
print(depth_norm)
depth_norm = depth_norm/max(depth_norm)
plt.title("depth graph")
plt.plot(x, depth_norm, color="black")
plt.show()

par_split = []
for i in range(len(depth_norm)):
    if(depth_norm[i] > 0.4):
      par_split.append(x[i])
print(par_split)

sentence_percentage = 0.2
last_split = 0
sim_score = np.sum(ss, axis=1)
sorted_indexes = [np.argsort(sim_score[0:par_split[0]])[::-1]]
for i in range(1, len(par_split)):
    to_app = np.argsort(sim_score[par_split[i-1]:par_split[i]])[::-1]
    to_app = to_app+par_split[i-1]
    sorted_indexes.append(to_app)
to_app = np.argsort(sim_score[par_split[len(par_split)-1]:])
to_app = to_app + par_split[len(par_split)-1]
sorted_indexes.append(to_app)

extracted = []
for l in sorted_indexes:
    extracted.append(np.sort(l[0:int(len(l)*sentence_percentage)]))
print(extracted)

from classla import Pipeline
classla.download('sr')
#classla.download('sr', type='nonstandard')
nlp = Pipeline(lang='sr', processors='tokenize,pos,lemma')

sentence = "Ovo je primer za lematizaciju."
doc = nlp(' '.join(get_sentences_from_file("tekst.txt")))
#doc = nlp(sentence)
print(doc)