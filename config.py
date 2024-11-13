import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4o-mini"


# Determine the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  

# Data Configuration
RAW_RECIPE_DATA_PATH = os.path.join(ROOT_DIR, "data", "Recipes.jsonl")

