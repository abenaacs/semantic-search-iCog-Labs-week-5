import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import streamlit as st
from dotenv import load_dotenv
import re
import os
import json


# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# Initialize Neo4j connection
@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


driver = get_neo4j_driver()

# Load dataset
file_path = "./data/flipkart_fashion_products_dataset.json"  # Path to your JSON file
df = pd.read_json(file_path).head(10000)


# Preprocess text
def preprocess_text(text):
    text = re.sub(r"\W", " ", text)  # Remove special characters
    return text.lower()  # Convert to lowercase


df["searchable_text"] = (
    df["title"] + " " + df["description"] + " " + df["brand"]
).apply(preprocess_text)

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Caching embeddings
if os.path.exists("embeddings_cache.json"):
    with open("embeddings_cache.json", "r") as f:
        df["embeddings"] = pd.Series(json.load(f))
else:
    df["embeddings"] = df["searchable_text"].apply(lambda x: model.encode(x).tolist())
    with open("embeddings_cache.json", "w") as f:
        json.dump(df["embeddings"].tolist(), f)

print("Embeddings computed and dataset preprocessed.")


# Query expansion using Neo4j
def query_expansion(query):
    synonyms = []
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Keyword {name: $query})-[:RELATED_TO]->(related)
            RETURN related.name AS synonym
            """,
            {"query": query.lower()},
        )
        synonyms = [record["synonym"] for record in result]
    return synonyms


# Search products with filters
def search_products_with_filters(query, category=None, tag=None, top_n=5):
    expanded_queries = [query] + query_expansion(query)
    query_embedding = model.encode(expanded_queries, convert_to_tensor=True)

    filtered_df = df
    if category:
        filtered_df = filtered_df[filtered_df["category"] == category]
    if tag:
        filtered_df = filtered_df[filtered_df["brand"].str.contains(tag, case=False)]

    product_embeddings = np.vstack(filtered_df["embeddings"].tolist())
    similarities = cosine_similarity(query_embedding, product_embeddings)
    mean_similarities = similarities.mean(axis=0)

    top_indices = mean_similarities.argsort()[-top_n:][::-1]
    return filtered_df.iloc[top_indices][["title", "description", "category", "brand"]]


# Streamlit App
st.title("Enhanced E-commerce Product Search")

# User input
query = st.text_input("Enter your search query:")
category = st.selectbox(
    "Filter by category:", ["All"] + df["category"].unique().tolist()
)
tag = st.text_input("Filter by tag (optional):")

if st.button("Search"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        # Search and display results
        try:
            results = search_products_with_filters(
                query, category=None if category == "All" else category, tag=tag
            )
            if results.empty:
                st.info("No results found.")
            else:
                st.write(f"Showing top {len(results)} results:")
                for idx, row in results.iterrows():
                    st.subheader(row["title"])
                    st.write(f"**Category**: {row['category']}")
                    st.write(f"**Description**: {row['description']}")
                    st.write(f"**Brand**: {row['brand']}")
                    st.write("---")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Close Neo4j connection explicitly when app reloads or stops
if st.button("Close Database Connection"):
    driver.close()
    st.write("Neo4j connection closed successfully.")
