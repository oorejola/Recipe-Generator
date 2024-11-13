import os
import sys
import pandas as pd
from typing import List, Dict, Tuple, Union
from langchain_openai import ChatOpenAI

try:
    from .recipe_type_utils import Recipe, Component, format_recipe
    from .vector_db import FAISSRecipeVectorStore, BM25RecipeVectorStore
except ImportError:
    # If relative imports fail, set up absolute imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from recipe_type_utils import Recipe, Component, format_recipe
    from vector_db import FAISSRecipeVectorStore, BM25RecipeVectorStore


# Add the project root to sys.path to ensure config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def hybrid_retriever(recipe_query:str,
                    FAISS_vector_store: FAISSRecipeVectorStore,
                    BM25_vector_store: BM25RecipeVectorStore,
                    k_FAISS = 4,
                    k_BMS= 4,
                    include_self=True)-> List[str]:
    """
    Retrieve the recipes of the most similar recipes to the query recipe using both FAISS and BM25.

    Args: recipe_query (str): The query recipe to search for.
            k_FAISS (int): The number of similar recipes to retrieve using FAISS.
            k_BMS (int): The number of similar recipes to retrieve using BM25.
            include_self (bool): Whether to include the query recipe in the search results or not.
    
    Returns: list of recipe names: The recipe names of the most similar recipes to the query recipe.
    """
    retrived_recipes_FAISS = FAISS_vector_store.retrieve_similar_recipes(recipe_query, k_FAISS, include_self=include_self)
    retrived_recipes_BM25 = BM25_vector_store.retrieve_similar_recipes(recipe_query, k_BMS, include_self=include_self)
    return list(set(retrived_recipes_FAISS+retrived_recipes_BM25))


class RAGRecipeGenerator:
    """
    A class for generating recipes using a Hybrid Retrieval-Augmented Generation (RAG) approach.
    The generator uses retrieved recipes to formulate a new recipe based on a provided query.
    """
    def __init__(self,
                df_recipes: pd.DataFrame,
                recipe_id_pairs: pd.DataFrame,
                completion_model= config.COMPLETION_MODEL,
                openai_api_key=config.OPENAI_API_KEY,
                embedding_model=config.EMBEDDING_MODEL
                ): 
        """
        Initializes the RAGRecipeGenerator with required resources such as the recipe DataFrame, vector stores, and OpenAI credentials.
        
        Args:
            df_recipes (pd.DataFrame): DataFrame containing recipes with columns like 'recipe', 'component', and 'amount'.
            recipe_id_pairs (pd.DataFrame): DataFrame mapping recipe IDs to their unique identifiers.
            completion_model (str): The model used for text generation (default from config).
            openai_api_key (str): API key for accessing OpenAI services.
            embedding_model (str): Model used for embedding (default from config).
        """
        # Store the recipe DataFrame and recipe ID pairs
        self.df_recipes = df_recipes
        self.recipe_id_pairs = recipe_id_pairs

        # Initialize the completion model, OpenAI API key, and embedding model
        self.competion_model=completion_model
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        
        # Initialize the FAISS and BM25 vector stores for recipe retrieval
        self.FAISS_vector_store = FAISSRecipeVectorStore(openai_api_key=self.openai_api_key,embedding_model=self.embedding_model)
        self.BM25_vector_store = BM25RecipeVectorStore()
        self.FAISS_vector_store.load_recipe_data(self.recipe_id_pairs)
        self.BM25_vector_store.load_recipe_data(self.recipe_id_pairs)
        
    def get_dict_of_components(self, recipe_name: str) -> List[Dict]:
        """
        Get components and their amounts for a specific recipe.
        Args:
            recipe_name (str): Name of the recipe to look up
        Returns:
            List[Dict]: List of dictionaries containing component names and amounts
        """
        recipe_component_dict = (
            self.df_recipes[self.df_recipes['recipe'] == recipe_name][['component', 'amount']]
            .to_dict(orient='records')
        )
        return recipe_component_dict
    
    def get_context(self, retrieved_recipes: List[str]) -> str:
        """
        Prepare the context (components and amounts) of the retrieved recipes to be used as input to the language model.
        
        Args:
            retrieved_recipes (List[str]): A list of recipe names to retrieve context for.

        Returns:
            str: A formatted string containing the components of the retrieved recipes.
        """
        context = []
        for recipe_name in retrieved_recipes:
            context.append(f"recipe_name = {recipe_name}")
            recipe_dict = self.get_dict_of_components(recipe_name)
            # Create Recipe and Component objects to format the recipe information
            recipe_object = Recipe(components=[
                Component(
                    component_name=component['component'],
                    ammount_of_component=round(component['amount'], 4)
                ) for component in recipe_dict
            ])
            context.append(str(recipe_object)) # Add the formatted recipe to the context
        return "\n".join(context)
    
    def generate_recipe_rag(
        self,
        recipe_query: str,
        k_FAISS: int = 4,
        k_BM25: int = 4,
        get_docs: bool = False,
        include_self: bool = False,
        temperature=0
    ) -> Union[List[Dict], Tuple[List[Dict], List[Dict]]]:
        """
        Generate a recipe using Hybrid RAG.
        
        Args:
            recipe_query (str): Query describing the desired recipe
            retriever: Retriever object for finding similar recipes
            k_FAISS (int): Number of recipes to retrieve using FAISS
            k_BM25 (int): Number of recipes to retrieve using BM25
            model (str): Name of the LLM model to use
            get_docs (bool): Whether to return retrieved recipes
            include_self (bool): Whether to include the query recipe in results if recovered via retriever
            
        Returns:
            Union[List[Dict], Tuple[List[Dict], List[Dict]]]: 
                Generated recipe and optionally retrieved recipes
        """

        # Retrieve similar recipes using hybrid retrieval (FAISS + BM25)
        retrieved_recipes = hybrid_retriever(
            recipe_query, 
            self.FAISS_vector_store,
            self.BM25_vector_store, 
            k_FAISS=k_FAISS, 
            k_BMS=k_BM25, 
            include_self=include_self
        )

        
        # Get the context for the retrieved recipes
        retrieved_context = self.get_context(retrieved_recipes)

        system_message = """You are an expert food scientist specializing in industrial recipe formulation.
            Task: Generate a recipe for {query} based on the similar recipes provided below.
            Similar Recipes:
            {context}
            Requirements:
            1. Use the reference recipes as inspiration but adapt amounts based on scientific principles
            2. All component amounts MUST sum to exactly 1 kg
            3. All component amounts MUST each be more than 0.005 kg
            4. Use standardized ingredient names (lowercase, singular form)
            5. Provide precise measurements (up to 4 decimal places)
            6. Ensure each component serves a specific functional purpose
            """
        
        system_message = system_message.format(
            query=recipe_query,
            context=retrieved_context
        )

        llm = ChatOpenAI(model=self.competion_model,openai_api_key=self.openai_api_key)
        structured_llm = llm.with_structured_output(Recipe, method="json_schema")
        generation_result = structured_llm.invoke(system_message, temperature=temperature)

        #Format Recipe from Recipe object to List of Dictionaries
        formatted_recipe = format_recipe(generation_result)
        if get_docs:
            return formatted_recipe, retrieved_recipes
        return formatted_recipe
    

def build_rag_recipe_generator():
    """
    Builds and returns an instance of the RAGRecipeGenerator using pre-loaded recipe data.
    Run pre_processing/data_loader.py to generate the cleaned_recipes.csv file before running this function.

    This function reads the cleaned recipe data from a CSV file, extracts the necessary recipe and UUID pairs, 
    and constructs an instance of the RAGRecipeGenerator for generating recipe suggestions based on a query.

    Returns:
        RAGRecipeGenerator: An instance of the RAGRecipeGenerator initialized with the cleaned recipe data.
    """
    # Load the cleaned recipe data
    cleaned_data_path = os.path.join(config.ROOT_DIR, "data", "cleaned_recipes.csv")
    df_recipes = pd.read_csv(cleaned_data_path, encoding='utf-8')
    
    # Prepare the recipe ID pairs (recipe name and associated UUID)
    recipe_id_pairs = df_recipes.drop_duplicates(subset=['recipe', 'uuid'])[['recipe', 'uuid']]
    
    # Create and return the RAGRecipeGenerator instance
    rag_recipe_generator = RAGRecipeGenerator(df_recipes, recipe_id_pairs)
    return rag_recipe_generator


if __name__ == "__main__":
    # Build the recipe generator
    hybrid_rag = build_rag_recipe_generator()
    # Define a recipe query
    recipe_query = "chocolate cake"

    # Generate the recipe based on the query
    generated_recipe = hybrid_rag.generate_recipe_rag(
        recipe_query,
        k_FAISS=4,
        k_BM25=4,
        get_docs=False
    )

    print(generated_recipe)