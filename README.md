# E‑Commerce Review Intelligence


 **NLP embeddings**, **t‑SNE exploration**, **prototype topic labels** (Quality / Fit / Style / Comfort), and **similar-review retrieval** with **Chroma** using OpenAI `text-embedding-3-small`.

## What it demonstrates

- **Data handling**: CSV discovery, null handling, reviews with non-empty text only  
- **Embeddings**: Batched OpenAI embedding calls  
- **EDA**: Rating and length distributions, department breakdowns, visual exports  
- **Unsupervised insight**: t‑SNE projection of embedding space  
- **Vector search**: Chroma persistent client (ingest, query, lifecycle)

## Prerequisites

- **Python 3.10+** recommended  
- **OpenAI API key** with access to embeddings  
- **Jupyter** (or VS Code / Cursor with a Jupyter kernel)

## Quick start

1. **Clone or copy** this folder and open a terminal in the project root.

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies** (aligned with the notebook):

   ```bash
   pip install "chromadb>=1.0.0" "openai>=1.0.0" pandas numpy scipy scikit-learn matplotlib python-dotenv
   ```

4. **Configure the API key** — create a `.env` file in the project root:

   ```bash
   OPENAI_API_KEY=sk-...
   ```

   The notebook loads this via `python-dotenv`. Do not commit `.env` (it is listed in `.gitignore`).

5. **Add the dataset** — the notebook looks for the [Women’s E‑Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) CSV under common paths, including:

   - `womens_clothing_e-commerce_reviews.csv` (project root or current working directory)  
   - `data/womens_clothing_e-commerce_reviews.csv`  
   - `~/Desktop/retail_reviews_data.csv`  

   Adjust the path in the notebook if your file lives elsewhere.

6. **Run the notebook**:

   ```bash
   jupyter notebook portfolio_project1_final.ipynb
   ```

   Run cells **top to bottom** the first time so imports, data loading, embeddings, and downstream cells stay in sync.

## Generated artifacts

Running the notebook writes figures to the project directory, for example:

- `eda_overview.png` — exploratory charts  
- `tsne_colored.png` — t‑SNE by topic and sentiment  
- `topic_summary.png` — business-intelligence style bars by topic  

Chroma also uses a **local persistent store** (path is set in the notebook); treat that folder as machine-local cache, not something to publish if it contains your data.

## Tech stack

| Layer        | Choice                          |
|-------------|----------------------------------|
| Language    | Python 3                         |
| Embeddings  | OpenAI `text-embedding-3-small`  |
| ML          | NumPy, SciPy, scikit-learn       |
| Viz         | Matplotlib                       |
| Vector DB   | ChromaDB                         |

## License

Notebook and code you add here are yours to license as you prefer. The original Kaggle dataset has its own terms on Kaggle.
