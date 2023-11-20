from rouge import Rouge

# Example reference text with three sentences
reference_text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves the development of algorithms and models to enable machines to understand, interpret, and generate human-like text. NLP plays a crucial role in various applications, including machine translation, sentiment analysis, and chatbot development.
"""

# Example hypothesis text with two sentences
hypothesis_text = "NLP is a branch of AI that deals with language processing. It has applications in machine translation and sentiment analysis."

# Create a Rouge object
rouge = Rouge()

# Compute ROUGE scores
scores = rouge.get_scores(hypothesis_text, reference_text, avg=False)

# Print the scores
print(scores)