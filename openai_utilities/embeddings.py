import os
from openai import OpenAI
from scipy import spatial
from .tokenutils import split_content
from enum import Enum

# the currently supported models in this code
class EmbeddingModels(Enum):
    # supported by anyscale.com
    # INPUT TOKEN LIMIT = 512.
    # Output vector length: 1024
    GTE_LARGE = "thenlper/gte-large" 
    # supported by openai.com
    # Input TOKEN LIMIT = 8191 
    # Output vector length: 1536
    ADA_002 = "text-embedding-ada-002" 

class EmbeddingAgent:
    def __init__(
            self,
            model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL"),
            api_key: str = None,
            organization: str = None,
            base_url: str = None):
        self.model = model
        self.openai_client = OpenAI(api_key=api_key, organization=organization, base_url=base_url)

    def __call__(self, input):
        return self.create(input)

    def create(self, text: str):
        # there is only 1 item the there will be only 1 item in the data array
        return self.openai_client.embeddings.create(input = text, model = self.model).data[0].embedding
    
    # chunks 1 large item with metadata_padding in consideration
    # the reason this function is split from the 1 below is that this way the embeddings can be batched
    def chunk_text(self, text: str, metadata_func = None):
        return split_content(text = text, model = self.model, metadata_func=metadata_func)
    
    # vector searches search_vector in the search_scope:
    # search_scope is an array items that will need to be search. this can be an array of vectors or array of objects that contains a field which represents the embeddings of the object
    # vector_item_func is the function that is used for extracting the embeddings for each item in the search scope
    # limit is the top number of items to return
    def search(self, query: str, search_scope, embeddings_item_func, limit: int = 1):
        search_vector = self.create(query)
        compare_func = lambda x, y: 1 - spatial.distance.cosine(x, y)    
        search_result = [(compare_func(search_vector, embeddings_item_func(item)), item) for item in search_scope]    
        search_result.sort(key = lambda x: x[0], reverse=True)
        _, items = zip(*search_result)
        return items[:limit]