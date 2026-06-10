from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
from backend.app.config import settings

class Embedder:
    def __init__(self):
        """
        Initialize the CodeBERT embedder.
        Uses GPU if available, otherwise falls back to CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Embedder on device: {self.device}")
        
        # We use microsoft/codebert-base as defined in settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.CODEBERT_MODEL)
        self.model = AutoModel.from_pretrained(settings.CODEBERT_MODEL).to(self.device)
        self.model.eval() # Set to evaluation mode

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Embed a list of text strings (code snippets or PR comments).
        Returns a list of dense vector embeddings.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, # CodeBERT max length
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use the [CLS] token embedding as the sentence representation
            # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize embeddings for cosine similarity
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
            
            # Convert to CPU list of floats
            batch_embeddings = cls_embeddings.cpu().numpy().tolist()
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]
