import os
import sys
import pandas as pd
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import tiktoken
from rank_bm25 import BM25Okapi



# Add the project root to sys.path to ensure config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class FAISSRecipeVectorStore:
    """
    A class to store and retrieve recipe data using FAISS (Facebook AI Similarity Search)
    and OpenAI embeddings. This class allows for efficient querying of similar recipes 
    based on a given input query by using embeddings for vector-based similarity search.
    """
    def __init__(self, openai_api_key=config.OPENAI_API_KEY, embedding_model=config.EMBEDDING_MODEL):
        """
        Initializes the FAISSRecipeVectorStore instance with the provided OpenAI API key and embedding model.

        Args:
            openai_api_key (str): The OpenAI API key to access the embedding model. Defaults to a configuration value.
            embedding_model (str): The name of the OpenAI model to use for generating embeddings. Defaults to a configuration value.
        """
                
        # Load the OpenAI API key from the .env file
        self.openai_api_key = openai_api_key
        self.recipe_id_pairs = None

        # Initialize the embedding model
        self.embedding_model = embedding_model
        self.embedder = OpenAIEmbeddings(model=self.embedding_model, openai_api_key=self.openai_api_key)
        
        # Determine embedding dimension by running a sample text
        sample_embedding = self.embedder.embed_query("sample text")
        self.embedding_dimension = len(sample_embedding)
        
        # Set up the FAISS index with the correct embedding dimension
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.vector_store = FAISS(
            embedding_function=self.embedder,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def load_recipe_data(self,recipe_id_pairs):
        """
        Loads recipe data into the FAISS vector store by generating embeddings and adding them to the FAISS index.

        Args:
            recipe_id_pairs (pd.DataFrame or dict): A data structure containing recipe IDs and recipe content.
                The 'recipe' column or key must contain the recipe text, and the 'uuid' column or key must contain unique IDs.
        """
        self.recipe_id_pairs = recipe_id_pairs
        data_recipes = recipe_id_pairs['recipe'].tolist()
        data_uuids = recipe_id_pairs['uuid'].tolist()
        documents = [Document(page_content=recipe_name) for recipe_name in data_recipes]
        self.vector_store.add_documents(documents=documents, ids=data_uuids)

    def retrieve_similar_recipes(self, query, k = 4,include_self = True):
        """
        Retrieves the top k most similar recipes to a given query using the FAISS index.

        Args:
            query (str): The query string to search for similar recipes.
            k (int): The number of similar recipes to retrieve. Default is 4.
            include_self (bool): If True, includes the query itself in the results if it is found. Default is True.

        Returns:
            list: A list of the top k most similar recipes to the input query, as decoded strings.
        """
        retrieved_docs = self.vector_store.similarity_search(query.lower(), k=k+1)
        retrieved_docs = [doc.page_content for doc in retrieved_docs]
        if include_self==True:
            return retrieved_docs[:k]
        else:
            if query in set(retrieved_docs):
                return retrieved_docs[1:]
            else:
                return retrieved_docs[:k]


class BM25RecipeVectorStore:
    """
    A class to store and retrieve recipe data using the BM25 algorithm for 
    information retrieval. This class tokenizes recipes and allows for efficient 
    querying of similar recipes based on a given input query.
    """
    def __init__(self):
        # Load the OpenAI API key from the .env file
        self.tokenizer = tiktoken.get_encoding("cl100k_base") 
        self.recipe_id_pairs = None
        self.tokenized_recipes = None
        self.bm25 = None

    def load_recipe_data(self,recipe_id_pairs):
        """
        Loads recipe data and tokenizes the recipes for use with BM25.

        Args:
            recipe_id_pairs (pd.DataFrame or dict): A data structure containing recipe IDs and the recipe content.
                The 'recipe' column or key must contain the recipe texts to be tokenized.
        """
        recipe_id_pairs = recipe_id_pairs
        data_recipes = recipe_id_pairs['recipe'].tolist()

        self.recipes_tokenized = [self.tokenizer.encode(recipe) for recipe in data_recipes]
        self.bm25 = BM25Okapi(self.recipes_tokenized)

    def retrieve_similar_recipes(self, query, k = 4,include_self = True):
        """
        Retrieves the top k most similar recipes to a given query using BM25.

        Args:
            query (str): The query string to search for similar recipes.
            k (int): The number of similar recipes to retrieve. Default is 4.
            include_self (bool): If True, includes the input query itself in the results if it's a match. 
                                  Default is True.

        Returns:
            list: A list of the top k most similar recipes to the input query, as decoded strings.
        """
        tokenized_query = self.tokenizer.encode(query.lower()) # Set query to lower case for better matching
        retrieved_docs = self.bm25.get_top_n(tokenized_query, self.recipes_tokenized, n=k)
        retrieved_docs = [self.tokenizer.decode(doc) for doc in retrieved_docs]

        if include_self==True:
            return retrieved_docs[:k]
        else:
            if query in set(retrieved_docs):
                return retrieved_docs[1:]
            else:
                return retrieved_docs[:k]



if __name__ == "__main__":
    cleaned_data_path = os.path.join(config.ROOT_DIR, "data", "cleaned_recipes.csv")

    df_recipes = pd.read_csv(cleaned_data_path, encoding='utf-8')
    recipe_id_pairs = df_recipes.drop_duplicates(subset=['recipe', 'uuid'])[['recipe', 'uuid']]

    # Test the FAISSRecipeVectorStore class
    recipe_vector_store = FAISSRecipeVectorStore()
    recipe_vector_store.load_recipe_data(recipe_id_pairs)

    # Test the vector store
    sample_query = 'chocolate pudding, with ice cream, and a cherry on top'
    similar_recipes = recipe_vector_store.retrieve_similar_recipes(sample_query, k=3)
    print("FAISS - Top 3 similar recipes:")
    for recipe in similar_recipes:
        print(recipe)

    # # Test the BM25RecipeVectorStore class
    recipe_vector_store = BM25RecipeVectorStore()
    recipe_vector_store.load_recipe_data(recipe_id_pairs)
    
    sample_query = 'chocolate pudding, with ice cream, and a cherry on top'
    similar_recipes = recipe_vector_store.retrieve_similar_recipes(sample_query, k=3)
    print("BM25 - Top 3 similar recipes:")
    for recipe in similar_recipes:
        print(recipe)


    
