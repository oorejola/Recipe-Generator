# Hybrid RAG Recipe Generation

This project is a Hybrid Retrieval-Augmented Generation (RAG) system designed to generate structured recipe outputs from unstructured user queries. By combining FAISS L2 Similarity for semantic matching and BM25 for term-based similarity, the system provides high-quality recipe suggestions that are both contextually accurate and textually relevant. Additionally, a LangChain Structured Query Output ensures that the generated recipes follow a reliable, schema-compliant structure.

### What does it do? 
Given a recipe query, we generate a list of dictionaries consisting of the components and their amounts that compose the recipe.

Example.
```python
recipe = hybrid_rag.generate_recipe("chocolate cake")
```
```
>>> [{'component': 'butter, 82% fat, unsalted', 'amount': 0.15}, {'component': 'chicken egg, raw, without shell', 'amount': 0.12}, {'component': 'dark chocolate', 'amount': 0.3}, {'component': 'semi-skimmed milk', 'amount': 0.1}, {'component': 'wheat flour', 'amount': 0.15}, {'component': 'white sugar', 'amount': 0.18}]
```
### How does it do it?
To do so, we create a sparse and dense vector database based on tokenization and embeddings of each recipe name in the dataset. Given a recipe query, we retrieve the most similar recipes. Then, with the retrieved recipes, we provide an LLM with the components and amounts of the retrieved recipes. Additionally, we provide clear instructions to the LLM as well as enforce structured query outputs.

### How does it preform?

To evaluate the performance of our method, we compute a series of metrics on a testing set. Given a training set as the database of recipes for which the hybrid-RAG model has access, the hybrid-RAG model is evaluated on multiple metrics that compare the hybrid-RAG generated recipe components against the true recipe components. Additionally, we compare the hybrid-RAG generated scores with those of a baseline model. The baseline model is simply a prompted LLM with the support of structured output. The full metrics can be found in `metrics/results.txt`, but here we highlight a few of the average performances.

| Metric   | Baseline   | Hybrid RAG  | Hybrid RAG > Baseline |
|------------|------------|------------|------------|
| LLM Oracle |  1.5769 | 2.2788 |0.54 |
| Cosine Sim | 0.7131 | 0.8073 | 0.85 |
| Jaccard Index | 0.0464 | 0.3439|    0.84|
| Fuzzy Jaccard Index | 0.2832 | 0.6197 | 0.91 |

The last column describes the proportion of times Hybrid RAG received a better score than the Baseline. We see that Hybrid RAG outperforms the baseline on all metrics.

### Data Cleaning and Preparation
The goal of this data processing step is to clean and normalize the recipe dataset, focusing on the components added in the recipes rather than those removed (e.g., water loss due to evaporation).
Here's a breakdown of the key steps:

1. Load Raw Data: The script reads the data/Recipe.jsonl file, which contains the raw recipe data, and converts it into a pandas DataFrame.
2. Text Standardization: The script uses a set of predefined patterns to remove unnecessary text and standardize the names of recipes and ingredients.
3. Remove Invalid Records: The script removes any records with zero or negative amounts, as well as duplicate records where the recipe name matches the ingredient name.
4. Filter Irrelevant Categories: The script removes any records belonging to the "processing" or "waste/emission" categories, as these are likely not relevant for the final analysis.
5. Standardize Units: The script ensures that all water-related ingredients are measured in kilograms (kg) rather than cubic meters (m3).
6. Normalize Ingredient Amounts: The script calculates the amount of each ingredient as a fraction of the total weight (in kg) for each recipe, making the values more comparable across recipes.
7. Add Unique Identifiers: The script assigns a unique identifier (UUID) to each recipe, which can be used as a primary key for the data.
8. Final Cleanup: The script ensures the recipe and ingredient names are in lowercase and aggregates the data by recipe and ingredient, summing the amounts where there are multiple entries for the same combination.

The end result is a cleaned and normalized DataFrame that can be used for further analysis or processing. This data cleaning process helps to ensure the data is in a consistent format, with invalid or irrelevant records removed, and the amounts standardized for better comparability.


## Project Structure

```
recipe_generator/
├── data/
│   └──  Recipes.jsonl # Dataset of recipes
├── preprocessing/
│   ├── __init__.py
│   └──  cleaner.py # Data cleaning utilities
├── metrics/
│   ├── __init__.py
│   ├── baseline_model.py
│   ├── evaluation.py   
│   └── results.txt # Results stored from last evaluation
├── rag/
│   ├── __init__.py
│   ├── rag.py # RAG pipeline for retrieval and generation
│   ├── recipe_type_utils.py    # Utilities for categorizing recipe types
│   └── vector_db.py # Vector database setup 
├── inference.py # Inference script for generating recipes
├── requirements.txt
├── config.py
└── README.md
```

## Features
1. Hybrid Retrieval
* *FAISS* for Semantic Similarity: The project utilizes Facebook AI Similarity Search (FAISS). Recipes are encoded as dense vectors, and FAISS enables efficient, high-speed L2 similarity search, retrieving recipes that are semantically similar to the input recipe.
* *BM25* for Keyword Matching: BM25 is used to retrieve documents based on term frequency and inverse document frequency. This method focuses on keyword and term-based relevance, retrieving recipes that closely match the specific words in the input recipe.
* *Hybrid Combination*: By combining FAISS and BM25, the system captures the strengths of both semantic similarity (contextual relevance) and textual similarity (keyword relevance).

2. Structured Output

* With *Pydantic*, and LangChain's *Structured Query Output*, this system enforces a consistent schema for generated recipes, ensuring each recipe includes standardized components within the `Recipe` and `Component` classes.

3. Evaluation
* Baseline Model Comparison: A baseline model is used to evaluate improvements in retrieval accuracy and recipe generation quality, providing a benchmark to gauge the added value of the hybrid retrieval approach.
* Evaluation Metrics:
    * LLM Oracle: A large language model (LLM) evaluates the similarity between the true and generated responses, scoring the generated recipe for accuracy and relevance.
    * Cosine Similarity: Measures the semantic similarity between embeddings of the true and generated responses, capturing the contextual closeness of the outputs.
    * Jaccard Index: Calculates the overlap between terms in the true and generated responses, assessing content similarity based on shared terms.
    * Fuzzy Jaccard Index: An enhanced Jaccard measure that averages directional Jaccard scores across subcomponents of each recipe, capturing nuanced content overlap.

## Prerequisites

This project was built with the following software and dependencies:

- Python version: 3.12.4
- Python packages:
  - See the `requirements.txt` file for a complete list of dependencies and their versions.

## Getting Started

1. Install the required dependencies.
```
pip install -r requirements.txt
```
2. Set the OpenAI API key in the `.env` file.
```
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

## Usage
1. Run the data preprocessing scripts.
```
python preprocessing/data_cleaning.py
```
2. Try it out by generating a recipe for `chocolate cake` with `inference.py`.
```
python inference.py ice cream sandwiches
```

3. To evaluate the preformance of the Hybrid RAG pipleline run:
```
python metrics/evaluation.py
```
Statistics are recorded in `results.txt`

## Example
To generate a recipe initialize and run `RAGRecipeGenerator':
```python
from rag.rag import build_rag_recipe_generator

hybrid_rag =  = build_rag_recipe_generator()
recipe = hybrid_rag.generate_recipe("chocolate cake")
print(recipe)
```
```
>>> [{'component': 'butter, 82% fat, unsalted', 'amount': 0.15}, {'component': 'chicken egg, raw, without shell', 'amount': 0.12}, {'component': 'dark chocolate', 'amount': 0.3}, {'component': 'semi-skimmed milk', 'amount': 0.1}, {'component': 'wheat flour', 'amount': 0.15}, {'component': 'white sugar', 'amount': 0.18}]
```
Output is a list of dictionaries.

## File Overview

* `data_processing/`: Handles data loading and cleaning. This prepares raw data into a format suitable for input into the retrieval and generation pipelines.
* `metrics/`: Contains baseline models (`baseline_model.py`) and evaluation program (`evaluation.py`). 
    * `baseline_model.py` consists of a structured recipe generation that does not make use of any retrieved information.
    * `evaluation.py` generates a test train split (0.15 test) RAG is preformed with access to recipes in the training set. For each recipe in the testing set 4 metrics (LLM as judge, Cosine Similarity, Jaccard Index, and Fuzzy Jaccard Distance) are computed for RAG geneneration and baseline model generation.
* `rag/`: The core of the retrieval-augmented generation, including:   
    * `vector_db.py` contains classes for sparse and dense vector database, BM25 and FAISS respectivly.
    * `rag.py` Main entry point for running recipe generation. Consists of the class `RAGRecipeGenerator` which acts as the model. `build_rag_recipe_generator` builds a `RAGRecipeGenerator` making use of the cleaned recipe data.
* `inference.py` is an easy access point to generate recipes from the comand line.
* `config.py` provides a centralized configuration for managing model parameters and files. Embedding model is defaulted to OpenAI's `"text-embedding-3-small"` and the Completion model (LLM) is defaulted to OpenAI's `"gpt-4o-mini"`



## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open a new issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
