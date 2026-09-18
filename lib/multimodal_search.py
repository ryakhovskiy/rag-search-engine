from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies
import pickle
from pathlib import Path
from dotenv import load_dotenv
from .semantic_search import cosine_similarity

load_dotenv()

class MultimodalSearch:

    __file_path = Path("cache/docs_clip.pkl")

    def __init__(self, model: str = "clip-ViT-B-32"):
        self.model = model
        self.transformer = SentenceTransformer(model)
        self.docs = load_movies()
        self.texts = [x['title'] + ': ' + x['description'] for x in self.docs]
        self.embeddings = self.get_embeddings_for_docs(self.texts)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        vector = self.transformer.encode(image, show_progress_bar=True)
        return vector

    def get_embeddings_for_docs(self, docs: list[str]):
        path = self.__file_path
        if path.exists():
            print(f"Loading embeddings from existing file: {path}")
            with open(path, "rb") as file:
                return pickle.load(file)
        else:
            print(f"Creating a new embeddings and saving to: {path}")
            embeddings = self.transformer.encode(docs, show_progress_bar=True)
            with open(path, "wb") as file:
                pickle.dump(embeddings, file)
            return embeddings

    def image_search(self, image_path: str):
        img_embedding = self.embed_image(image_path)
        csimilarities = []
        for i, emb in enumerate(self.embeddings, start=0):
            cs = cosine_similarity(emb, img_embedding)
            csimilarities.append({'title': self.docs[i]['title'], 'description': self.docs[i]['description'][:100], 'cs': cs})
        return sorted(csimilarities, key=lambda x: x['cs'], reverse=True)[:5]

def verify_image_embedding(image_path: str):
    print(f"verifying image embedding for the image '{image_path}'")
    mms = MultimodalSearch()
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def search_using_image(image_path: str):
    print(f"searching data based on the image '{image_path}'")
    mms = MultimodalSearch()
    results = mms.image_search(image_path)
    for i, res in enumerate(results, start=1):
        print(f"{i}. {res['title']} (similarity: {res['cs']:.3f})\n   {res['description']}\n")
