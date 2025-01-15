# E-commerce Product Search Tool

This repository contains an advanced search tool for e-commerce datasets, combining semantic search, query expansion, and a knowledge graph integration to enhance product discovery and recommendation capabilities. The tool uses Streamlit for the frontend, SentenceTransformer for embedding-based semantic search, and Neo4j for query expansion using a knowledge graph.

---

## Features

1. **Semantic Search**: Leverages Sentence-BERT (SBERT) to provide accurate product recommendations based on natural language queries.
2. **Knowledge Graph Integration**: Enhances search accuracy by identifying relationships between entities in queries using Neo4j.
3. **Query Expansion**: Suggests synonyms or related terms using the knowledge graph to refine search results.
4. **Efficient Embedding Caching**: Optimizes speed by caching computed embeddings locally.
5. **Filters**: Allows users to filter results by category and brand.
6. **Interactive Frontend**: Built with Streamlit, providing a user-friendly interface.

---

## Installation

### Prerequisites

- Python 3.8 or above
- Neo4j (Community or Enterprise Edition)
- Pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/abenaacs/semantic-search-iCog-Labs-week-5.git
cd semantic-search-iCog-Labs-week-5
```

### Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Neo4j Configuration

1. Download and install Neo4j: [Neo4j Download](https://neo4j.com/download/)
2. Start the Neo4j server and log in to the Neo4j Browser.
3. Import the knowledge graph data:
   - Open the Neo4j browser and run the provided Cypher commands to create nodes and relationships.

---

## Usage

### Running the Application

1. Ensure Neo4j is running.
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the URL displayed in the terminal (e.g., `http://localhost:7687`).

### Application Workflow

1. Enter a search query (e.g., "Men's cotton track pants").
2. Optionally filter results by category or brand.
3. View recommended products with detailed descriptions and brands.

---

## Dataset

The application uses a dataset containing over 1.9 million rows of e-commerce product information. To optimize performance, the dataset is trimmed to the first 10,000 rows. The dataset includes the following fields:

- `_id`: Unique product identifier
- `title`: Product title
- `description`: Product description
- `brand`: Product brand
- `category`: Product category
- `sub_category`: Product sub-category
- `images`: Image URLs
- `actual_price`: Original price
- `selling_price`: Discounted price
- `discount`: Discount percentage
- `average_rating`: Customer rating

---

## Knowledge Graph Integration

The knowledge graph uses Neo4j to model relationships between entities, such as:

- `Keyword`: Represents search terms.
- `RELATED_TO`: Relationship between keywords and their synonyms or related concepts.

### Sample Cypher Queries

1. Add a keyword:
   ```cypher
   CREATE (n:Keyword {name: 'track pants'})
   ```
2. Relate two keywords:
   ```cypher
   MATCH (a:Keyword {name: 'track pants'}), (b:Keyword {name: 'cotton'})
   CREATE (a)-[:RELATED_TO]->(b)
   ```
3. Query synonyms:
   ```cypher
   MATCH (n:Keyword {name: 'track pants'})-[:RELATED_TO]->(related)
   RETURN related.name
   ```

---

## File Structure

```
.
├── app.py                  # Main Streamlit application
├── data/
│   └── flipkart_fashion_products_dataset.json
├── embeddings_cache.json   # Cached embeddings
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

---

## Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Key Libraries

- **Streamlit**: Interactive UI for the search tool
- **SentenceTransformers**: Embedding-based semantic search
- **Neo4j**: Knowledge graph database
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **scikit-learn**: Cosine similarity computations

---

## Performance Optimization

- **Trimming Dataset**: Reduced to the first 10,000 rows for faster processing.
- **Embedding Caching**: Saved embeddings locally to minimize recomputation.

---

## Future Enhancements

1. **Scalability**: Implement distributed computing for larger datasets.
2. **Advanced Filtering**: Add multi-level filters for finer search granularity.
3. **Real-time Knowledge Graph Updates**: Dynamically add new nodes and relationships.
4. **Multi-language Support**: Extend search capabilities for non-English queries.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions or improvements.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For questions or suggestions, please contact:

- **Name**: Abenezer Nigussie
- **Email**: abenezernigussiecs@gmail.com
- **GitHub**: [abenaacs](https://github.com/abenaacs)

---

## Acknowledgments

- [Neo4j](https://neo4j.com/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [Flipkart Dataset](#)

Thank you for exploring this project!
