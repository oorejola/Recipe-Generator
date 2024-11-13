from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd

class Component(BaseModel):
    """A component of a recipe with normalized measurements"""
    component_name: str = Field(
        ..., 
        description="Name of the ingredient"
    )
    ammount_of_component: float = Field(
        ..., 
        description="Amount of the component in kg"
    )

class Recipe(BaseModel):
    """A recipe with components and their amounts"""
    components: List[Component] = Field(
        ...,
        description="List of recipe components that must sum to 1 kg"
    )

def format_recipe(result: Recipe) -> List[Dict]:
        """
        Format recipe into standardized output format.
        
        Args:
            result (Recipe): Recipe object to format
            
        Returns:
            List[Dict]: List of dictionaries containing component information
        """
        return [
            {
                'component': component.component_name,
                'amount': component.ammount_of_component
            }
            for component in result.components
        ]

def get_dict_of_components(recipe_name: str, df_recipes:pd.DataFrame) -> List[Dict]:
        """
        Get components and their amounts for a specific recipe.
        
        Args:
            recipe_name (str): Name of the recipe to look up
            
        Returns:
            List[Dict]: List of dictionaries containing component names and amounts
        """
        recipe_component_dict = (
            df_recipes[df_recipes['recipe'] == recipe_name][['component', 'amount']]
            .to_dict(orient='records')
        )
        return recipe_component_dict