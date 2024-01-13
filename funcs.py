import classla
from classla import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import re
import json
import math
def get_sentances_and_sums_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    sentences = []
    sums = []
    for entry in data:
        source_sentences = entry["source"]
        sentences.append(source_sentences)
    for entry in data:
        source_sum = entry["target"]
        sums.append(source_sum)
    return (sentences, sums)

def get_sentences_from_file(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        pattern = r'\.{3}\n|\.{3}|\.\n|\. |[!] |[?] |[!]+\n|[?]+\n'
        sentences = re.split(pattern, text)
        sentences = [sentence.replace("\n", " ") for sentence in sentences]
    return sentences

def lematize_list_of_sentences(sentences, nlp):
    sentences_lem = []
    doc = nlp(' '.join(sentences))
    for sent in doc.sentences:
        sentences_lem.append(" ".join([word.lemma for word in sent.words]))
    #print(sentences_lem)
    return sentences_lem
def lematize_matrix_of_sentences(matrix, nlp):
    sentences_lem = []
    ret_matrix = []
    for list in matrix:
        ret_matrix.append(lematize_list_of_sentences(list, nlp))
    return ret_matrix

def remove_zero_columns(X):
    # Find indices of columns with all zeros
    non_zero_columns = np.any(X != 0, axis=0)

    # Create a new matrix without columns of all zeros
    return X[:, non_zero_columns]

def countvec(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    X = X.toarray()
    X = np.transpose(X)
    X = remove_zero_columns(X)
    return X

def tfidf(sentences):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    X = X.toarray()
    X = np.transpose(X)
    X = remove_zero_columns(X)
    return X
    #a = list(vectorizer.get_feature_names_out())
def reduce(Um, Sm, Vt, l):
    sum = np.sum(Sm)
    for i in range(int(len(Sm)/2), len(Sm)):
      if(i == len(Sm)-1):
          return(Um, Sm, Vt)
      if(np.sum(Sm[:i])/sum >= l):#ulazi ako prvih i vrednosti zadovoljava nejdnakost
        return (np.delete(Um,np.s_[i:],1), Sm[:i], np.delete(Vt, np.s_[i:], 0))

def WW(Um, Sm):
  return (Um * Sm**2) @ np.transpose(Um)

def SS(Vt, Sm):
    ss = (np.transpose(Vt) * Sm**2) @ Vt
    return ss

def calculate_adjacent_similarity(ss):
    adj_sim = np.diagonal(ss, offset=1)
    return adj_sim

def depth_old(ss, paragraph_split_treshold = 1):
    adj_sim_norm = calculate_adjacent_similarity(ss)
    x = []
    depth = [1]
    for i in range(1, len(adj_sim_norm) - 1):
        if adj_sim_norm[i - 1] > adj_sim_norm[i] and adj_sim_norm[i] < adj_sim_norm[i + 1]:
            adj_sim_norm_i = adj_sim_norm[i]
            if adj_sim_norm[i] == 0: adj_sim_norm_i = 0.000000000000000000001
            depth.append((adj_sim_norm[i-1]+adj_sim_norm[i+1])/(adj_sim_norm_i*2)-1)
            x.append(i)
    paragraph_split = []
    depth = np.array(depth)
    depth = depth/ np.max(depth)
    for i in range(1, len(depth)):
        if(depth[i] > paragraph_split_treshold):
          paragraph_split.append(x[i-1])
    return [depth, paragraph_split]

def depth_improved_function(ss, paragraph_split_treshold=0.4):
    adj_sim_norm = calculate_adjacent_similarity(ss)
    x = []
    depth_norm = [1]
    min = []
    prev_l_max = adj_sim_norm[0]
    prev_r_max = adj_sim_norm[len(adj_sim_norm)-1]
    l_max = []
    r_max = []
    for i in range(1, len(adj_sim_norm) - 1):
        if adj_sim_norm[i - 1] > adj_sim_norm[i] and adj_sim_norm[i] < adj_sim_norm[i + 1]:
            x.append(i-1)
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
    depth_norm = np.array(depth_norm)
    depth_norm = depth_norm/np.max(depth_norm)
    paragraph_split = []
    for i in range(1, len(depth_norm)):
        if(depth_norm[i] > paragraph_split_treshold):
          paragraph_split.append(x[i-1]+1)
    return [depth_norm, paragraph_split]

def extract_sentances(X, sentence_percentage, depth_function, paragraph_split_treshold = 0.5):
    Um, Sm, Vt = np.linalg.svd(X)
    (Um, Sm, Vt) = reduce(Um, Sm, Vt, 0.90)
    ss = SS(Vt, Sm)
    par_split = depth_function(ss, paragraph_split_treshold)[1]
    sim_score = np.sum(ss, axis=1)
    if(len(par_split) != 0):
        sorted_indexes = [np.argsort(sim_score[0:par_split[0]])[::-1]]
        for i in range(1, len(par_split)):
            to_app = np.argsort(sim_score[par_split[i-1]:par_split[i]])[::-1]
            to_app = to_app+par_split[i-1]
            sorted_indexes.append(to_app)
        to_app = np.argsort(sim_score[par_split[len(par_split)-1]:])[::-1]
        to_app = to_app + par_split[len(par_split)-1]
        sorted_indexes.append(to_app)
    else:
        to_app = np.argsort(sim_score)[::-1]
        sorted_indexes = [to_app]

    extracted = []
    for l in sorted_indexes:
        extracted = np.concatenate((extracted, (np.sort(l[0:math.ceil(len(l)*sentence_percentage)]))))
    extracted = extracted.astype(int)
    return extracted

def average_alfa(X, depth_function, n):
    Um, Sm, Vt = np.linalg.svd(X)
    (Um, Sm, Vt) = reduce(Um, Sm, Vt, 0.95)
    ss = SS(Vt, Sm)
    depth = np.array(depth_function(ss, 1)[0]) 
    depth = np.sort(depth)[::-1]
    return (depth[n-1] + depth[n])/2   