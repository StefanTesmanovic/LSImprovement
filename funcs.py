import classla
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def get_sentences_from_file(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        pattern = r'\.{3}\n|\.{3}|\.\n|\. |[!] |[?] |[!]+\n|[?]+\n'
        sentences = re.split(pattern, text)
        sentences = [sentence.replace("\n", " ") for sentence in sentences]
    return sentences

def remove_zero_columns(X):
    # Find indices of columns with all zeros
    non_zero_columns = np.any(X != 0, axis=0)

    # Create a new matrix without columns of all zeros
    return X[:, non_zero_columns]
def tfidf(file_name):
    vectorizer = TfidfVectorizer()
    file_name = "tekst.txt"
    #X = vectorizer.fit_transform(corpus)
    X = vectorizer.fit_transform(get_sentences_from_file(file_name))
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
    return (np.transpose(Vt) * Sm**2) @ Vt

def depth(ss):
    pass
def calculate_adjacent_similarity(ss):
    intensities = np.linalg.norm(ss, axis=0)
    ss_norm = ss/intensities
    adj_sim_norm = np.diagonal(ss_norm, offset=1)
    return adj_sim_norm

def depth_improved_function(ss, paragraph_split_treshold):
    adj_sim_norm = calculate_adjacent_similarity(ss)
    x = []
    depth_norm = []
    min = []
    prev_l_max = adj_sim_norm[0]
    prev_r_max = adj_sim_norm[len(adj_sim_norm)-1]
    l_max = []
    r_max = []
    for i in range(1, len(adj_sim_norm) - 1):
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
    depth_norm = depth_norm/max(depth_norm)
    paragraph_split = []
    for i in range(len(depth_norm)):
        if(depth_norm[i] > paragraph_split_treshold):
          paragraph_split.append(x[i])
    return [depth_norm, paragraph_split]

def extract_sentances(ss, sentence_percentage, depth_function, paragraph_split_treshold):
    par_split = depth_function(ss, paragraph_split_treshold)[1]
    intensities = np.linalg.norm(ss, axis=0)
    ss_norm = ss/intensities
    sim_score = np.sum(ss_norm, axis=1)
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
        extracted = np.concatenate((extracted, (np.sort(l[0:int(len(l)*sentence_percentage)]))))
    extracted = extracted.astype(int)
    return extracted
    