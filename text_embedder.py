from transformers import CLIPModel, CLIPTokenizer
import torch

class TextEmbedder():

    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_to_embeddings = {}
    
    def _embed_text(self,string):
        input_ids = self.tokenizer(string, return_tensors="pt",padding=True).input_ids
        with torch.no_grad():
            text_features= self.model.get_text_features(input_ids=input_ids).detach()[0]
        return text_features
        
    def embed_text(self,text):
        if text not in self.text_to_embeddings:
            self.text_to_embeddings[text] = self._embed_text(text)
        return self.text_to_embeddings[text]