# The project explanation in English
  
  The project's goal is to improve an extractive text summarization method called LSI (Latent Semantic Indexing). LSI utilizes Singular Value Decomposition (SVD) to decompose and perform dimensionality reduction on the word-sentence count matrix. Due to dimensionality reduction, the calculation time is reduced, and only the most relevant information is preserved. The outcome of SVD is an approximation matrix with rank k (where k < n, and n is the rank of the original matrix) that has the smallest possible Euclidean distance from the original matrix. The improvement consists of using the tf-idf metric instead of the word-sentence matrix, changing the paragraphing formula, and preprocessing the text by lemmatizing it. 
  
  The original method starts with creating the word-sentence count matrix and performing the SVD on it. Afterwards it calculates the sentence similarities and splits the text into paragraphs using the similarity values. The last step is to extract the chosen number of sentences with the highest similarity scores from each paragraph.
  
  Lemmatization was chosen because words in Serbian have many forms. Without lemmatization, every form would be considered a distinct word, which is not accurate and increases the dimensions of the word-sentence matrix.
  
  The change in the paragraphing formula was chosen because the original formula considers only one sentence before and one after the candidate sentences for text division. The new formula solves this by considering all sentences in the text. The calculation is optimized by passing through the text only once and reusing the data.
  
  The reason for choosing tf-idf instead of a simple word-sentence count matrix is that the words that appear most frequently in the text are the least important for the meaning (for example, words like 'a,' 'the,' 'for,' 'of').

Word-sentence count matrix example. 
Every row represents a sentence and every column represents how many times does that word appear in every sentence.

  ![image](https://github.com/StefanTesmanovic/LSImprovement/assets/83782548/7ba80315-4a86-4a4b-836e-d2976f75bf96)
