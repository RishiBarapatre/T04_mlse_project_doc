# Multimodal AI Pokedex (RAG + Vision)

A local, multimodal Retrieval-Augmented Generation (RAG) system that functions as an intelligent Pokedex. This project allows users to search for Pokémon using natural language descriptions (Text-to-Image) and identify Pokémon by uploading images (Image-to-Text), utilizing a hybrid database of images and stats.

## Features

* **Text-to-Image Search (Semantic Retrieval):**
    * Search for Pokémon using abstract descriptions (e.g., "A fire dragon," "Pink balloon").
    * Uses **OpenCLIP** to understand the semantic meaning of queries.
    * Returns the **Top 3** visual matches to handle ambiguity (e.g., distinguishing between Charizard and other red Pokémon).
    * Displays the Pokémon's image along with stats like Name and Type.

* **Image-to-Text Identification (Visual Search + Vision AI):**
    * Upload any image of a Pokémon.
    * Uses **Pixel-based Visual Search** (via embeddings) to identify the specific Pokémon from the database with near-100% accuracy.
    * Uses **Moondream** (a small Vision Language Model running via Ollama) to generate a unique description of the uploaded image (e.g., describing pose, expression, or color).

* **Local & Efficient:**
    * Runs entirely on a standard laptop CPU (tested on Ryzen 5500U).
    * No expensive GPUs required.
    * Uses **ChromaDB** for efficient vector storage and retrieval.

## 🛠️ Tech Stack

* **Language:** Python
* **Database:** [ChromaDB](https://www.trychroma.com/) (Vector Store)
* **Embeddings:** OpenCLIP (via `open_clip_torch`)
* **Vision Model:** Moondream (via [Ollama](https://ollama.com/))
* **Data Handling:** Pandas (CSV processing), Pillow (Image processing), Numpy

## 📂 Project Structure

```text
Pokedex_Project/
├── images/               # Folder containing Pokemon images (e.g., bulbasaur.png)
├── pokemon.csv           # Dataset containing stats (Name, Type1, Type2, Evolution)
├── pokedex.py            # Main application script
└── README.md             # This file
```

How It Works

    Indexing: The system reads the pokemon.csv and matches rows to images in the images/ folder. It uses OpenCLIP to convert every image into a mathematical vector (embedding) and stores it in ChromaDB.

    Retrieval:

        Text Query: The user's text is converted to a vector. ChromaDB finds the images with the closest vector distance (cosine similarity).

        Image Query: The uploaded image's pixels are converted to a vector. ChromaDB finds the exact visual match in the database.

    Generation: For image queries, the Moondream model analyzes the visual features to provide a human-readable description.
