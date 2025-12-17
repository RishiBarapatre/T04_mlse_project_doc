import os
import pandas as pd
import numpy as np
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import ollama
from tkinter import filedialog
import tkinter as tk
from huggingface_hub import InferenceClient

IMAGE_FOLDER = "images"       
CSV_FILE = "pokemon.csv"      
COLLECTION_NAME = "pokedex_hybrid"

HF_TOKEN = "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"

print("Powering up the Multimodal Pokedex...")

embedding_func = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()
client = chromadb.Client()

try: client.delete_collection(COLLECTION_NAME)
except: pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    data_loader=data_loader
)

print("📊 Reading CSV and matching with Images...")
try:
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
except:
    print(f"Error: Could not read {CSV_FILE}")
    exit()

if 'name' not in df.columns:
    print("Critical Error: CSV must have a 'Name' column.")
    exit()

ids = []
image_paths = []
metadatas = []

for index, row in df.iterrows():
    name = str(row['name']).strip().lower()
    
    img_path_png = os.path.join(IMAGE_FOLDER, f"{name}.png")
    img_path_jpg = os.path.join(IMAGE_FOLDER, f"{name}.jpg")
    
    final_path = None
    if os.path.exists(img_path_png):
        final_path = img_path_png
    elif os.path.exists(img_path_jpg):
        final_path = img_path_jpg
    
    if final_path:
        ids.append(name)
        image_paths.append(final_path)
        metadatas.append({
            "name": str(row['name']),
            "type1": str(row.get('type1', 'Unknown')), 
            "type2": str(row.get('type2', '')),
            "evolution": str(row.get('evolution', 'None'))
        })

if not ids:
    print("No matches found! Check file naming.")
    exit()

print(f"   Found {len(ids)} Pokemon. Indexing...")
collection.add(ids=ids, uris=image_paths, metadatas=metadatas)
print("Pokedex Ready!")

def text_to_image():
    """Type text -> Get Top 3 Candidates"""
    query = input("\nDescribe the Pokemon: ")
    print("Searching Visual Database...")
    
    results = collection.query(
        query_texts=[query], n_results=3, include=['uris', 'metadatas']
    )
    
    if not results['uris'] or not results['uris'][0]:
        print("Nothing found.")
        return

    print(f"\nTop 3 Matches for '{query}':")
    candidates = []
    for i in range(len(results['uris'][0])):
        name = results['metadatas'][0][i]['name']
        img_path = results['uris'][0][i]
        types = f"{results['metadatas'][0][i]['type1']} {results['metadatas'][0][i]['type2']}"
        candidates.append(img_path)
        print(f"   {i+1}. {name} ({types})")

    selection = input("\nWHICH ONE? (Type 1, 2, or 3): ")
    if selection in ['1', '2', '3']:
        idx = int(selection) - 1
        Image.open(candidates[idx]).show()

def image_to_text():
    """Upload Image -> Get Identification + Description"""
    print("\nSelect a Pokemon image to analyze...")
    root = tk.Tk()
    root.withdraw()
    query_path = filedialog.askopenfilename()
    
    if not query_path: return
    print(f"👀 Analyzing {os.path.basename(query_path)}...")
    
    try:
        query_image = Image.open(query_path).convert("RGB")
        query_array = np.array(query_image)
    except:
        print("Error loading image.")
        return

    print("🔍 Comparing pixels with Pokedex database...")
    results = collection.query(
        query_images=[query_array], n_results=1, include=['metadatas']
    )
    
    if not results['metadatas'] or not results['metadatas'][0]:
        print("No match found.")
        return

    match_data = results['metadatas'][0][0]
    print(f"\nIDENTIFIED: It is {match_data['name'].upper()}!")
    
    print("\n   Asking Vision AI for observations...")
    try:
        vision_response = ollama.chat(
            model='moondream',
            messages=[{'role': 'user', 'content': f"Describe this image of {match_data['name']}.", 'images': [query_path]}]
        )
        print(f"Vision Analysis: {vision_response['message']['content']}")
    except:
        pass

while True:
    print("\n" + "="*30)
    print("1. Text-to-Image (Search)")
    print("2. Image-to-Text (Identify)")
    print("q. Quit")
    
    choice = input("> ")
    if choice == '1': text_to_image()
    elif choice == '2': image_to_text()
    elif choice == 'q': break