import streamlit as st
import os
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import math
from collections import Counter

############################
# Functions from your script
############################

def load_documents(folder_path):
    documents = {}
    doc_id = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                documents[doc_id] = {'filename': filename, 'content': text}
                doc_id += 1
    return documents

def preprocess_text(text, language='english'):
    text = text.lower()
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator)
    tokens = text.split()
    stop_words = stopwords.words(language)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_tokens = {}
    for doc_id, doc in documents.items():
        tokens = preprocess_text(doc['content'])
        doc_tokens[doc_id] = tokens
        for token in tokens:
            inverted_index[token][doc_id] += 1
    return inverted_index, doc_tokens

def compute_tfidf(inverted_index, doc_tokens, N):
    tfidf = defaultdict(dict)
    idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf[term] = math.log(N / (1 + df))
    for doc_id, tokens in doc_tokens.items():
        term_counts = Counter(tokens)
        max_tf = max(term_counts.values())
        for term, count in term_counts.items():
            tf = count / max_tf
            tfidf[doc_id][term] = tf * idf[term]
    return tfidf, idf

def compute_query_tfidf(query, idf):
    query_tokens = preprocess_text(query)
    term_counts = Counter(query_tokens)
    max_tf = max(term_counts.values()) if len(term_counts) > 0 else 1
    query_tfidf = {}
    for term, count in term_counts.items():
        tf = count / max_tf
        query_tfidf[term] = tf * idf.get(term, 0)
    return query_tfidf

def cosine_similarity(doc_vector, query_vector):
    common_terms = set(doc_vector.keys()).intersection(set(query_vector.keys()))
    dot_product = sum(doc_vector[t] * query_vector[t] for t in common_terms)
    doc_norm = math.sqrt(sum(value**2 for value in doc_vector.values()))
    query_norm = math.sqrt(sum(value**2 for value in query_vector.values()))
    if doc_norm == 0 or query_norm == 0:
        return 0.0
    else:
        return dot_product / (doc_norm * query_norm)

def search(query, tfidf, idf, top_k=5):
    query_vector = compute_query_tfidf(query, idf)
    scores = {}
    for doc_id, doc_vector in tfidf.items():
        scores[doc_id] = cosine_similarity(doc_vector, query_vector)
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs[:top_k]

############################
# Streamlit App
############################

nltk.download('stopwords')

st.title("Document Search Engine")

# Upload folder path
folder_path = st.text_input("Enter the folder path containing text files:", "./documents")

if folder_path and os.path.exists(folder_path):
    documents = load_documents(folder_path)
    N = len(documents)

    if N > 0:
        st.success(f"Loaded {N} documents.")

        # Build inverted index
        inverted_index, doc_tokens = build_inverted_index(documents)
        tfidf, idf = compute_tfidf(inverted_index, doc_tokens, N)

        query = st.text_input("Enter your search query:")

        if query:
            results = search(query, tfidf, idf, top_k=5)

            if results:
                st.write(f"Top {len(results)} results for '{query}':")
                for rank, (doc_id, score) in enumerate(results, start=1):
                    doc_info = documents[doc_id]
                    st.markdown(f"### Rank {rank}: {doc_info['filename']}")
                    st.write(f"**Score:** {score:.4f}")
                    with st.expander("View Content"):
                        st.write(doc_info['content'])
            else:
                st.warning("No results found.")
    else:
        st.error("No documents found in the specified folder.")
else:
    st.error("Please provide a valid folder path.")
