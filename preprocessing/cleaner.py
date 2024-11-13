import os
import sys
from uuid import uuid4
import json
import pandas as pd

# Add the project root to sys.path to ensure config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_jsonl(file_path)->pd.DataFrame:
    """Load and transform JSONL data into a pandas DataFrame."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            item_name = item['name']
            for component in item['recipe']:
                component['recipe'] = item_name
                data.append(component)
    
    df = pd.DataFrame(data)
    df = df[['recipe', 'amount', 'unit', 'name', 'category']]
    df.rename(columns={'name': 'component'}, inplace=True)
    return df

class RecipeCleaner:
    """
    A class for cleaning and processing recipe data, including tasks like removing invalid records,
    standardizing units, normalizing amounts, and assigning unique identifiers.
    """
    def __init__(self, input_file_path , output_file_path):
        """
        Initialize the RecipeCleaner with input and output file paths.

        Parameters:
            input_file_path (str): Path to the raw recipe data file.
            output_file_path (str): Path where the cleaned data will be saved.
        """
        self.input_file= input_file_path
        self.output_file= output_file_path
        self.patters_to_replace=  [
        (r' \{[A-Za-z]+\} U', ''),
        (r',? at [^,]+|, at .+', ''),
        (r',?\s*to be reheated', ''),
        (r'\bw\b', 'with'),
        (r',?\s*for processing', ''),
        (r',?\s*consumption mix', ''),
        (r',?\s*to be filled', ''),
        (r',?\s*production', ''),
        (r', for [^,]+|, for .+$', ''),
        (r'^\[Dummy\]\s*', '') 
    ]
        
    def clean_text_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text patterns in recipe and component names by removing unnecessary substrings.
        """
        for pattern, replacement in self.patters_to_replace:
            df['component'] = df['component'].str.replace(pattern, replacement, regex=True)
            df['recipe'] = df['recipe'].str.replace(pattern, replacement, regex=True)
        return df
    
    def remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid or redundant records, specifically those with zero or negative amounts,
        and duplicate records in certain categories.
        """
        df = df[~((df['recipe'] == df['component'])&(df['category'] == 'material/Agricultural/Food/Recipes'))]
        df = df[df['amount'] != 0]
        df = df[df['amount'] >= 0.00000001]
        return df
    
    def remove_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows belonging to unnecessary categories, specifically 'processing' and 'waste/emission' categories.
        """
        # Identify category types
        processing_categories = [cat for cat in df['category'].unique() if 'processing' in cat]
        waste_categories = [cat for cat in df['category'].unique() 
                          if 'waste' in cat or 'Emission' in cat]
        df = df[~df['category'].isin(processing_categories + waste_categories)]
        return df
    
    def standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize units, converting 'm3' to 'kg' specifically for water.
        """
        df.loc[df['unit'] == 'm3', 'amount'] *= 1000
        df.loc[df['unit'] == 'm3', 'unit'] = 'kg'
        return df
    
    def standardize_water_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize all water-related components to a single label, 'water'.
        """
        water_mask = df['component'].str.contains('water, municipal', case=False, na=False)
        df.loc[water_mask, 'component'] = 'water'
        water_mask = df['component'].str.contains('tap water', case=False, na=False)
        df.loc[water_mask, 'component'] = 'water'
        return df

    def normalize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize amounts based on the total weight for each recipe in kilograms.
        """
        # Then, assign values based on a condition using .loc to avoid SettingWithCopyWarning
        df.loc[:, 'amount'] = df['amount'] / df[df['unit'] == 'kg'].groupby('recipe')['amount'].transform('sum')
        return df
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete cleaning pipeline, including text pattern cleaning, removal of invalid records,
        category filtering, unit standardization, and amount normalization.

        Parameters:
            df (DataFrame): The DataFrame containing raw recipe data.

        Returns:
            DataFrame: Fully cleaned and processed DataFrame ready for analysis or storage.
        """
        df = self.clean_text_patterns(df)
        df = self.remove_invalid_records(df)
        df = self.remove_categories(df)
        df = self.standardize_units(df)
        df = self.standardize_water_components(df)
        df = self.normalize_amounts(df)
        
        # Final cleanup
        df = df[['recipe', 'component', 'amount', 'unit']]
        df['recipe'] = df['recipe'].str.lower()
        df['component'] = df['component'].str.lower()
        
        # Aggregate final results
        return df.groupby(['recipe', 'component'], as_index=False)['amount'].sum()
    
    def add_uuid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a unique identifier (UUID) to each record.
        """
        uuid_index_map = {}
        for recipe_name in df['recipe'].unique():
            uuid = str(uuid4()) 
            uuid_index_map[recipe_name] = uuid
        df['uuid'] = df['recipe'].map(uuid_index_map)
        return df

if __name__ == "__main__":

    # Load raw data
    data_path = config.RAW_RECIPE_DATA_PATH
    cleaned_data_path = os.path.join(config.ROOT_DIR, "data", "cleaned_recipes.csv")
    print(f"Loading data from {data_path}...")
    df = load_jsonl(data_path)

    # Clean data
    print("Cleaning data...")
    cleaner = RecipeCleaner(input_file_path=data_path, output_file_path=cleaned_data_path)
    cleaned_df = cleaner.clean(df)

    print("Assigning unique identifiers...")
    cleaned_df = cleaner.add_uuid(cleaned_df)

    # Save results
    print(f"Saving cleaned data to {cleaned_data_path}...")
    cleaned_df.to_csv(cleaned_data_path, index=False)
    print(f"Total Cleaned Recipes: {len(cleaned_df['recipe'].unique())}")

   