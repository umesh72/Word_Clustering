from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

def assign_cluster_names(clusters):
    cluster_names = {}
    
    # Iterate through each cluster and extract keywords
    for cluster, terms in clusters.items():
        # Concatenate all terms in the cluster into a single string
        cluster_text = ' '.join(terms)
        
        # Use TF-IDF vectorizer to extract keywords
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([cluster_text])
        feature_names = vectorizer.get_feature_names_out()
        word_scores = np.array(X.sum(axis=0))[0]
        
        # Extract top 2 highest scoring words as keywords
        top_keywords = [feature_names[idx] for idx in np.argsort(word_scores)[::-1][:2]]
        
        # Generate a name for the cluster based on keywords
        cluster_names[cluster] = " ".join(top_keywords).capitalize()
    
    return cluster_names

def cluster_terms_agglomerative(terms, min_clusters=2, max_clusters=10, linkage='ward'):
    # Convert terms to lowercase
    terms_lower = [term.lower() for term in terms]

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform terms to TF-IDF features
    X = vectorizer.fit_transform(terms_lower)

    best_score = -1
    best_n_clusters = -1

    # Iterate over different numbers of clusters
    for n_clusters in range(min_clusters, max_clusters+1):
        # Perform Agglomerative clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        agg_clustering.fit(X.toarray())

        # Get cluster labels
        cluster_labels = agg_clustering.labels_

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

        # Update best silhouette score and number of clusters if necessary
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters

    print(f"\nBest number of clusters: {best_n_clusters} (Silhouette Score: {best_score})")

    # Perform clustering with the best number of clusters
    agg_clustering = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=linkage)
    agg_clustering.fit(X.toarray())

    # Get cluster labels
    cluster_labels = agg_clustering.labels_

    # Print clusters
    unique_labels = np.unique(agg_clustering.labels_)
    clusters = {i: [] for i in unique_labels}
    for i, term in enumerate(terms):
        cluster = agg_clustering.labels_[i]
        clusters[cluster].append(term)
    
    # Assign names to clusters based on keywords
    cluster_names = assign_cluster_names(clusters)

    # Print assigned names for each cluster and cluster data
    for cluster, name in cluster_names.items():
        print(f"\n{name}:")
        for term in clusters[cluster]:
            print(f"   - {term}")

# Input terms
terms = [ "Electrical", "Non-Electrical", "Mechnaical", 
  "Chemical ","political" 
]

# Define range of clusters to test
min_clusters = 2
max_clusters = 20

# Perform clustering and select the best number of clusters
cluster_terms_agglomerative(terms, min_clusters, max_clusters)
