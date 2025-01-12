import os
import math
import string
import nltk
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Pour la racinisation
# Pour le français, on peut utiliser 'FrenchStemmer' ou 'SnowballStemmer("french")'
# from nltk.stem.snowball import FrenchStemmer

# Assurez-vous d'avoir téléchargé les stopwords NLTK:
# nltk.download('stopwords')

############################
# 1. Chargement des textes #
############################

def load_documents(folder_path):
    """
    Charge tous les documents texte d'un dossier et renvoie un dictionnaire
    {doc_id: contenu_texte}.
    """
    documents = {}
    doc_id = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                documents[doc_id] = text
                doc_id += 1
    return documents

############################
# 2. Prétraitement         #
############################

def preprocess_text(text, language='english'):
    """
    Prétraite le texte : 
    - Met en minuscules
    - Supprime la ponctuation
    - Supprime les stopwords
    - (Optionnel) applique un stemmer
    """
    # Passage en minuscules
    text = text.lower()
    
    # Suppression de la ponctuation
    # On peut remplacer chaque signe de ponctuation par un espace
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator)

    # Tokenisation basique (split sur l'espace)
    tokens = text.split()
    
    # Chargement des stopwords
    stop_words = stopwords.words(language)
    
    # Racinisation (anglais). Pour le français, utiliser SnowballStemmer("french") par exemple.
    stemmer = PorterStemmer()
    
    # Filtrage des stopwords et racinisation
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    return tokens

############################
# 3. Index inversé         #
############################

def build_inverted_index(documents):
    """
    Construit un index inversé au format :
    {
        "terme": { doc_id: tf_dans_le_doc, ... },
        ...
    }
    et renvoie également la liste des tokens par document au format:
    {
        doc_id: [liste de tokens],
        ...
    }
    """
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_tokens = {}
    
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)  # prétraitement
        doc_tokens[doc_id] = tokens
        for token in tokens:
            inverted_index[token][doc_id] += 1  # fréquence du token dans le doc_id

    return inverted_index, doc_tokens

############################
# 4. Calcul des TF-IDF     #
############################

def compute_tfidf(inverted_index, doc_tokens, N):
    """
    Calcule les poids tf-idf pour chaque terme dans chaque document.
    Renvoie une structure:
    {
        doc_id: { "terme": tfidf, ... },
        ...
    }
    où N est le nombre total de documents.
    """
    tfidf = defaultdict(dict)
    
    # Calcul IDF pour chaque terme
    # idf(term) = log(N / (1 + nombre_de_docs_qui_contiennent_le_terme))
    idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)  # nb de documents contenant 'term'
        idf[term] = math.log(N / (1 + df))
        
    # Pour chaque doc, calcul du TF-IDF
    for doc_id, tokens in doc_tokens.items():
        # Compter la fréquence de chaque token dans ce doc
        term_counts = Counter(tokens)
        max_tf = max(term_counts.values())  # pour normaliser, optionnel
        for term, count in term_counts.items():
            # TF simple : count / max_tf (ou juste count)
            tf = count / max_tf
            tfidf[doc_id][term] = tf * idf[term]
            
    return tfidf, idf

############################
# 5. Recherche             #
############################

def compute_query_tfidf(query, idf):
    """
    Calcule le vecteur tf-idf de la requête.
    La requête est prétraitée avant le calcul.
    """
    query_tokens = preprocess_text(query)
    term_counts = Counter(query_tokens)
    max_tf = max(term_counts.values()) if len(term_counts) > 0 else 1
    
    # Vecteur TF-IDF de la requête
    query_tfidf = {}
    for term, count in term_counts.items():
        tf = count / max_tf
        # idf.get(term, 0) si le terme n'existe pas dans l'index, on considère IDF = 0
        query_tfidf[term] = tf * idf.get(term, 0)
        
    return query_tfidf

def cosine_similarity(doc_vector, query_vector):
    """
    Calcule la similarité cosinus entre deux vecteurs sous forme de dictionnaires :
    doc_vector = {term: tfidf_value, ...}
    query_vector = {term: tfidf_value, ...}
    """
    # Produit scalaire
    common_terms = set(doc_vector.keys()).intersection(set(query_vector.keys()))
    dot_product = sum(doc_vector[t] * query_vector[t] for t in common_terms)
    
    # Norme du document
    doc_norm = math.sqrt(sum(value**2 for value in doc_vector.values()))
    # Norme de la requête
    query_norm = math.sqrt(sum(value**2 for value in query_vector.values()))
    
    if doc_norm == 0 or query_norm == 0:
        return 0.0
    else:
        return dot_product / (doc_norm * query_norm)

def search(query, tfidf, idf, top_k=5):
    """
    Recherche les documents les plus pertinents pour la requête.
    Renvoie les doc_ids triés par score de similarité décroissante.
    """
    query_vector = compute_query_tfidf(query, idf)
    
    # Calcul de la similarité pour chaque doc
    scores = {}
    for doc_id, doc_vector in tfidf.items():
        scores[doc_id] = cosine_similarity(doc_vector, query_vector)
    
    # Tri des documents par ordre décroissant de score
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_docs[:top_k]

############################
# Programme principal      #
############################
if __name__ == "__main__":
    # 1) Charger les documents (assurez-vous d'avoir un dossier "documents" contenant 100 fichiers .txt)
    folder_path = "./documents"
    documents = load_documents(folder_path)
    N = len(documents)
    
    # 2 & 3) Construire l'index inversé
    inverted_index, doc_tokens = build_inverted_index(documents)
    
    # 4) Calculer le TF-IDF
    tfidf, idf = compute_tfidf(inverted_index, doc_tokens, N)
    
    # 5) Effectuer des recherches
    while True:
        user_query = input("\nEntrez votre requête (ou tapez 'exit' pour quitter) : ")
        if user_query.lower() == 'exit':
            print("Fin du programme.")
            break
        
        results = search(user_query, tfidf, idf, top_k=5)
        print(f"\nRésultats pour la requête : '{user_query}'")
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"#{rank} | Document ID: {doc_id} | Score: {score:.4f}")
