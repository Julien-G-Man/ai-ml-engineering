"""
Classification with embeddings

1. Zero shot classification:
  - Not using labeled data

Process:
  - Embed class descriptions
  - Embed the item to classify
  - Calculate distances
  - Assign the most similar label
"""

from scipy.spatial import distance
from embeddings import create_embeddings
from semantic_search import create_article_text

topics = [
    {'label': 'Tech',     'description': 'A news article about technology'},
    {'label': 'Science',  'description': 'A news article about Science'},
    {'label': 'Sport',    'description': 'A news article about sports'},
    {'label': 'Business', 'description': 'A news article about business'}
]

article =  {
    "headline": "How Nvidia GPUs could decide Who Wins the AI Race",
    "topic":    "Tech",
    "keywords": ["ai", "business", "computers"]
}

def classify():
    class_descriptions = [topic['description'] for topic in topics]
    class_embeddings = create_embeddings(class_descriptions)

    article_text = create_article_text(article)
    article_embeddings = create_embeddings(article_text)[0]
    
    closest = find_closest(article_embeddings, class_embeddings)
    label = topics[closest['index']]['label']
    return label
    

def find_closest(query_vector, embeddings):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    return min(distances, key=lambda x: x["distance"])


if __name__ == "__main__":
    label = classify()
    print(label)