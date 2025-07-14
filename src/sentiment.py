# src/sentiment.py

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class TextEncoder:
    def __init__(
        self,
        model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        cache_dir: str = "./models"
    ):
        # Tokenizer pour prétraiter le texte
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        # Modèle BERT de base (sans tête de classification)
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir
        )

    def extract_features(self, texts: list[str]) -> torch.Tensor:
        """
        Prend en entrée une liste de chaînes et renvoie
        les embeddings du token [CLS] pour chaque texte.
        """
        # 1. Tokenisation
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        # 2. Passage dans le modèle sans calcul de gradient
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 3. Extraction de l'embedding du token [CLS]
        return outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

    def analyze_sentiment(text: str) -> dict:
        """
        Analyse le sentiment d'un texte avec un modèle déjà fine-tuned
        (nlptown/bert-base-multilingual-uncased-sentiment) et renvoie
        un dict {label: confidence}.
        """
        # Chargement du tokenizer et du modèle de classification
        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment", cache_dir="./models"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment", cache_dir="./models"
        )

        # Préparation
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        # Les classes vont de 1 à 5, on choisit la plus probable
        label_idx = int(torch.argmax(torch.tensor(probs))) + 1
        if label_idx <= 2:
            label = "négatif"
        elif label_idx == 3:
            label = "neutre"
        else:
            label = "positif"
        confidence = round(max(probs), 3)
        return {label: confidence}

