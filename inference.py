import argparse
import pandas as pd
import os
from rag.rag import RAGRecipeGenerator
from rag.rag import build_rag_recipe_generator
import config


# def build_rag_recipe_generator():
#     """
#     Builds and returns an instance of the RAGRecipeGenerator using pre-loaded recipe data.
#     Run pre_processing/data_loader.py to generate the cleaned_recipes.csv file before running this function.

#     This function reads the cleaned recipe data from a CSV file, extracts the necessary recipe and UUID pairs, 
#     and constructs an instance of the RAGRecipeGenerator for generating recipe suggestions based on a query.

#     Returns:
#         RAGRecipeGenerator: An instance of the RAGRecipeGenerator initialized with the cleaned recipe data.
#     """
#     # Load the cleaned recipe data
#     cleaned_data_path = os.path.join(config.ROOT_DIR, "data", "cleaned_recipes.csv")
#     df_recipes = pd.read_csv(cleaned_data_path, encoding='utf-8')
    
#     # Prepare the recipe ID pairs (recipe name and associated UUID)
#     recipe_id_pairs = df_recipes.drop_duplicates(subset=['recipe', 'uuid'])[['recipe', 'uuid']]
    
#     # Create and return the RAGRecipeGenerator instance
#     rag_recipe_generator = RAGRecipeGenerator(df_recipes, recipe_id_pairs)
#     return rag_recipe_generator

# Set up the argument parser
def main():
    """
    Main function that handles command-line input, builds the RAGRecipeGenerator, 
    and generates a recipe based on the user's query.
    
    This function uses argparse to parse the query from the command line, 
    initializes the recipe generator, and then prints the recipe suggestion 
    generated for the input query.

    Usage:
        python inference.py "recipe query"
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate a recipe for a given query.")
    parser.add_argument("query", type=str, nargs='+', help="Recipe query to generate a recipe (supports spaces and special characters)")
    args = parser.parse_args()
    
    # Join all parts of the query in case it was split into multiple parts
    query = ' '.join(args.query)

    
    # Build the recipe generator
    rag_recipe_generator = build_rag_recipe_generator()
    
    # Generate a recipe for the query
    recipe = rag_recipe_generator.generate_recipe_rag(query)  
    
    for item in recipe:
        print(f"{item['component']} {item['amount']} kg")
    

if __name__ == "__main__":
    main()
