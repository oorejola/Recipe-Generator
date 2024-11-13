from typing import List, Dict
import os 
import sys
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from baseline_model import BaselineGenerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rag.rag import RAGRecipeGenerator



class Evaluator:
    """
    A class to evaluate the accuracy of generated recipes against true recipes using various metrics.
    """
    def __init__(self,
                recipe_query: str,
                generated_recipe: List[Dict],
                df_recipes: pd.DataFrame,
                completion_model= config.COMPLETION_MODEL,
                openai_api_key=config.OPENAI_API_KEY,
                embedding_model=config.EMBEDDING_MODEL
                ):
        """
        Initialize the Evaluator with the required parameters.

        Args:
            recipe_query (str): The name or description of the recipe.
            generated_recipe (List[Dict]): List of components for the generated recipe.
            df_recipes (pd.DataFrame): DataFrame containing true recipe components.
            completion_model (str): The model used for evaluation (default from config).
            openai_api_key (str): API key for accessing OpenAI services.
            embedding_model (str): Model used for embedding (default from config).
        """

        self.generated_recipe = generated_recipe
        self.recipe_query = recipe_query
        self.df_recipes = df_recipes
        self.completion_model = completion_model
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model

        self.true_recipe = self.get_dict_of_components(recipe_query)

    def pretty_recipe(self,recipe):
        """ Takes a generated recipe in structured format and returns a pretty string """
        pretty_string = []
        for component in recipe:
            pretty_string.append(f"{component['component']} {round(component['amount'],4)} kg")

        return '\n'.join(pretty_string)
    
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
    
    def oracle_evaluate_generated_response(self) -> int:
        """
        Evaluate the generated recipe's accuracy on a scale of 0-5 using a language model.

        Returns:
            int: The evaluation score from 0 to 5, indicating accuracy.
        """
        system_message_judge = """
        Your task is to evaluate the accuracy of the generated recipe compared to the true recipe for {query} on a scale of 0-5, where:

        0 = Completely inaccurate with no matching components
        1 = <20% matching components or approximate amounts
        2 = 20-40% matching components or approximate amounts
        3 = 40-60% matching components or approximate amounts
        4 = 60-80% matching components or approximate amounts
        5 = >80% matching components or approximate amounts

        When scoring, consider the following criteria:
        - Different names for the same components (e.g.,"sodium chloride" vs "salt", "phyllo" vs "filo") should be treated as matching
        - Ignore ignore information in the true recipe that is not directly related to the recipe components (e.g., instructions, serving size)
        - Exact amounts do not need to match, but similar proportions are important
        - Focus on the accuracy of the overall recipe composition

        Generated Recipe: {generated}

        True Recipe: {true}

        Respond ONLY with a single number (0-5) representing the accuracy score."""

        system_message_judge = system_message_judge.format(
            query=self.recipe_query,
            generated=self.pretty_recipe(self.generated_recipe),
            true=self.pretty_recipe(self.true_recipe)
        )

        llm = ChatOpenAI(model = self.completion_model , openai_api_key= self.openai_api_key)
        
        evaluation_result = llm.invoke(system_message_judge, temperature=0,max_tokens=1)
        return evaluation_result.content[0]
    
    def cosine_similarity(self) -> float:
        """
        Compute the cosine similarity between the generated and true recipes.
        0 indicates no similarity, 1 indicates identical recipes.

        Returns:
            float: The cosine similarity score between 0 and 1.
        """
        # Initialize the embedding model
        embeddings = OpenAIEmbeddings(
            model=self.embedding_model,openai_api_key=self.openai_api_key
        )
        
        # Get embeddings for both texts
        generated_recipe_embedding = embeddings.embed_query(self.pretty_recipe(self.generated_recipe))
        true_recipe_embedding = embeddings.embed_query(self.pretty_recipe(self.true_recipe))
        
        # Calculate cosine similarity
        similarity = np.dot(generated_recipe_embedding, true_recipe_embedding) / (norm(generated_recipe_embedding) * norm(true_recipe_embedding))
        return similarity
    

    def get_ingredients(self,recipe: List[Dict]) -> set:
        """
        Extract the ingredients (components) from a recipe.

        Args:
            recipe (List[Dict]): The recipe to extract ingredients from.

        Returns:
            set: A set of component names.
        """
        components = [component['component'] for component in recipe]
        return set(components)
    

    def jaccard_index(self)-> float:
        """
        Calculate the Jaccard Index for comparing ingredients in the generated and true recipes. 
        0 indicates no similarity, 1 indicates identical recipe components.

        Returns:
            float: The Jaccard similarity score between 0 and 1.
        """
        generated_recipe_set = self.get_ingredients(self.generated_recipe)
        true_recipe_set = self.get_ingredients(self.true_recipe)
        return len(generated_recipe_set.intersection(true_recipe_set)) / len(generated_recipe_set.union(true_recipe_set))
    
    def fuzzy_directional_jaccard_index(self,recipe_1: List[dict], recipe_2: List[dict]):
        """
        Compute a fuzzy directional Jaccard Index considering ingredient subcomponents.

        Args:
            recipe_1 (List[Dict]): The first recipe.
            recipe_2 (List[Dict]): The second recipe.

        Returns:
            float: The fuzzy directional Jaccard similarity.
        """
        recipe_set_1 = self.get_ingredients(recipe_1)
        recipe_set_2 = self.get_ingredients(recipe_2).copy()  # Copy to safely modify recipe_set_2

        match_count = 0

        for ingredient_1 in recipe_set_1:
            sub_ingredients = ingredient_1.split(', ')

            # Check for any matching sub-ingredient in recipe_set_2
            match = next((i2 for i2 in recipe_set_2 if any(sub in i2 for sub in sub_ingredients)), None)
            if match:
                match_count += 1
                recipe_set_2.remove(match)  # Remove matched ingredient to avoid re-use

        # Calculate the directional Jaccard Index
        return match_count / len(recipe_set_1) if recipe_set_1 else 0

    def fuzzy_jaccard_index(self):
        """
        Compute a fuzzy Jaccard Index considering ingredient subcomponents. 
        0 indicates no similarity, 1 indicates identical recipe subcomponents.

        Args:
            recipe_1 (List[Dict]): The first recipe.
            recipe_2 (List[Dict]): The second recipe.

        Returns:
            float: The fuzzy directional Jaccard similarity.
        """
        index_1_to_2 = self.fuzzy_directional_jaccard_index(self.generated_recipe, self.true_recipe)
        index_2_to_1 = self.fuzzy_directional_jaccard_index( self.true_recipe, self.generated_recipe)
        return (index_1_to_2 + index_2_to_1) / 2



if __name__=="__main__":
    cleaned_data_path = os.path.join(config.ROOT_DIR, "data", "cleaned_recipes.csv")
    df_recipes = pd.read_csv(cleaned_data_path, encoding='utf-8')
    recipe_id_pairs = df_recipes.drop_duplicates(subset=['recipe', 'uuid'])[['recipe', 'uuid']]

    # Split the data into training and testing sets
    train_recipe_id_pairs, test_recipe_id_pairs = train_test_split(recipe_id_pairs, test_size=0.15, random_state=30)
    print(f'Training set size: {len(train_recipe_id_pairs)}')
    print(f'Testing set size: {len(test_recipe_id_pairs)}')

    # Create the RAG Recipe Generator with the training data
    print('Building the RAG Recipe Generator...')
    rag_recipe_generator = RAGRecipeGenerator(df_recipes,train_recipe_id_pairs)

    # Create the Baseline Recipe Generator
    baseline_generator = BaselineGenerator()

    scores = []
    count =0
    for _, row in test_recipe_id_pairs.iterrows():
        recipe_query = row['recipe']
        recipe_uuid = row['uuid']
        print(f'Generating recipes for query: {recipe_query} ---  ({count+1}/{len(test_recipe_id_pairs)})')
        rag_recipe = rag_recipe_generator.generate_recipe_rag(recipe_query,k_FAISS=6,k_BM25=6,get_docs=False)
        baseline_recipe = baseline_generator.generate_recipe_baseline(recipe_query)

        print('Scoring the generated recipes...')
        rag_evaluator = Evaluator(recipe_query=recipe_query,df_recipes=df_recipes, generated_recipe=rag_recipe)
        baseline_evaluator = Evaluator(recipe_query=recipe_query,df_recipes=df_recipes, generated_recipe=baseline_recipe)

        rag_score_oracle = rag_evaluator.oracle_evaluate_generated_response()
        baseline_score_oracle = baseline_evaluator.oracle_evaluate_generated_response()
        print(f'RAG Recipe Score: {rag_score_oracle}')
        print(f'Baseline Recipe Score: {baseline_score_oracle}')

        rag_score_cossim = rag_evaluator.cosine_similarity()
        baseline_score_cossim = baseline_evaluator.cosine_similarity()
        print(f'RAG Recipe CosSim: {rag_score_cossim}')
        print(f'Baseline Recipe CosSim: {baseline_score_cossim}')

        rag_score_jaccard_index= rag_evaluator.jaccard_index()
        baseline_score_jaccard_index= baseline_evaluator.jaccard_index()
        print(f'RAG Recipe Jaccard Index: {rag_score_jaccard_index}')
        print(f'Baseline Recipe Jaccard Index: {baseline_score_jaccard_index}')

        rag_score_fuzzy_jaccard_index= rag_evaluator.fuzzy_jaccard_index()
        baseline_score_fuzzy_jaccard_index= baseline_evaluator.fuzzy_jaccard_index()
        print(f'RAG Recipe Fuzzy Jaccard Index: {rag_score_fuzzy_jaccard_index}')
        print(f'Baseline Recipe Fuzzy Jaccard Index: {baseline_score_fuzzy_jaccard_index}')
        score ={'recipe_query':recipe_query, 
                'rag_score_oracle':rag_score_oracle,
                'baseline_score_oracle':baseline_score_oracle,
                'rag_score_cossim':rag_score_cossim,
                'baseline_score_cossim':baseline_score_cossim,
                'rag_score_jaccard_index':rag_score_jaccard_index,
                'baseline_score_jaccard_index':baseline_score_jaccard_index,
                'rag_score_fuzzy_jaccard_index':rag_score_fuzzy_jaccard_index,
                'baseline_score_fuzzy_jaccard_index':baseline_score_fuzzy_jaccard_index,
                'recipe_uuid':recipe_uuid}
        scores.append(score)
        count+=1

    scores_df = pd.DataFrame(scores)
    scores_df['rag_score_oracle'] = pd.to_numeric(scores_df['rag_score_oracle'], errors='coerce')
    scores_df['baseline_score_oracle'] = pd.to_numeric(scores_df['baseline_score_oracle'], errors='coerce')


    # Save the evaluation results to a file
    evaluation_data_path = os.path.join(config.ROOT_DIR, "metrics", "results.txt")
    original_stdout = sys.stdout

    # Open file and redirect stdout
    with open(evaluation_data_path, 'w') as f:
        sys.stdout = f
        
        # Your existing print statements stay exactly the same
        print("\n\n\n")
        print(f'Training set size: {len(train_recipe_id_pairs)}')
        print(f'Testing set size: {len(test_recipe_id_pairs)}')
        
        stats = scores_df.describe()
        print('Evaluation metrics:')
        # Loop through each column's statistics
        for column in stats.columns:
            print(f'\n{column}')
            for stat, value in stats[column].items():
                print(f'  {stat}: {value}')  
            
        means = stats.loc['mean']
        print("\n")
        print("Average scores:")
        for column, mean_value in means.items():
            print(f"{column}: {mean_value:.4f}")
            
        print("\n\n\n")
        print("Proportion of RAG scores higher than baseline (oracle):")
        print((scores_df['rag_score_oracle'] > scores_df['baseline_score_oracle']).mean())
        print("Proporation of RAG scores higher than baseline (cosine similarity):")
        print((scores_df['rag_score_cossim'] > scores_df['baseline_score_cossim']).mean())
        print("Proportion of RAG scores higher than baseline (jaccard index):")
        print((scores_df['rag_score_jaccard_index'] > scores_df['baseline_score_jaccard_index']).mean())
        print("Proportion of RAG scores higher than baseline (fuzzy jaccard index):")
        print((scores_df['rag_score_fuzzy_jaccard_index'] > scores_df['baseline_score_fuzzy_jaccard_index']).mean())

    # Restore stdout
    sys.stdout = original_stdout