import os
import sys
import pandas as pd
from typing import List, Dict, Tuple, Union
from langchain_openai import ChatOpenAI
# Add the project root to sys.path to ensure config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rag.recipe_type_utils import Recipe, Component, format_recipe

class BaselineGenerator():
    """
    A class to generate baseline recipe formulations based on provided recipe queries,
    using language models without additional retrieval-augmented generation (RAG).
    """
    def __init__(self,
                completion_model= config.COMPLETION_MODEL,
                openai_api_key=config.OPENAI_API_KEY):
        self.completion_model = completion_model
        self.openai_api_key = openai_api_key

    def generate_recipe_baseline(self, recipe_query: str,temperature=0) -> List[Dict]:
        """
        Generate a basic recipe without using RAG or additional context.
        
        Args:
            recipe_query (str): The name of the recipe to generate
            
        Returns:
            List[Dict]: List of recipe components with names and amounts
        """
        
        system_message = f"""You are an expert food scientist specializing in industrial recipe formulation.
    Task: Generate a recipe for {recipe_query}.
    Requirements:
    1. Use the reference recipes as inspiration but adapt amounts based on scientific principles
    2. All component amounts MUST sum to exactly 1 kg
    3. Use standardized ingredient names (lowercase, singular form)
    4. Provide precise measurements (up to 4 decimal places)
    5. Ensure each component serves a specific functional purpose

    """
        
        llm = ChatOpenAI(model=self.completion_model,openai_api_key=self.openai_api_key)
        structured_llm = llm.with_structured_output(Recipe, method="json_schema")
        generation_result =structured_llm.invoke(system_message,temperature=temperature)

        #Format Recipe from Recipe object to List of Dictionaries
        formatted_recipe = format_recipe(generation_result)

        return formatted_recipe
    
if __name__=="__main__":
    generator = BaselineGenerator()
    recipe = generator.generate_recipe_baseline("goblet of ice cream, coffee or chocolate ice cream topped with whipped cream")
    print(recipe)