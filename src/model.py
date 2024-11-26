from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ThemeClassifier:
    def __init__(self, themes):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.themes = themes
        self.theme_embeddings = self.model.encode(themes)
    
    def classify(self, question, threshold=0.5):
        question_embedding = self.model.encode([question])
        similarities = cosine_similarity(question_embedding, self.theme_embeddings)[0]
        best_match_index = np.argmax(similarities)
        
        return self.themes[best_match_index] if similarities[best_match_index] > threshold else "Unknown"
      
      
if __name__ == "__main__":
    themes = ['Cashless', 'Fraud', 'Cards', 'Online Shopping']
    classifier = ThemeClassifier(themes)

    print(classifier.classify("ինչ է ձկնորսությունը?"))  
    print(classifier.classify("What is fishing?"))  