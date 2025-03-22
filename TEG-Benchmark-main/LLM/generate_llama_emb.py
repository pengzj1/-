import argparse
import pickle
import numpy as np
import torch
import os
import requests
import time
from datasets import Dataset
from tqdm import tqdm
import requests

class OllamaEmbeddingGenerator:
    def __init__(self, model_name="llama3:8b", batch_size=10, max_retries=3, retry_delay=2):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_url = "http://127.0.0.1:11434/api/embeddings"
        # Check if Ollama is running
        try:
            response = requests.get("http://127.0.0.1:11434/api/version")
            print(f"Connected to Ollama version: {response.json().get('version')}")
        except Exception as e:
            print(f"Error connecting to Ollama API: {e}")
            print("Please make sure Ollama is running on http://127.0.0.1:11434")
            raise

    def get_embedding_dimension(self):
        """Get the embedding dimension from a test query"""
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": "test"}
            )
            response.raise_for_status()
            return len(response.json().get("embedding", []))
        except Exception as e:
            print(f"Failed to determine embedding dimension: {e}")
            # Fallback dimension for Llama3-8b
            return 4096

    def get_embedding(self, text):
        """Get embedding for a single text with retries"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={"model": self.model_name, "prompt": text}
                )
                response.raise_for_status()
                return response.json().get("embedding")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Retry {attempt+1}/{self.max_retries}. Error: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to get embedding after {self.max_retries} attempts: {e}")
                    return None

    def get_batch_embeddings(self, texts):
        """Process texts in batches to avoid overloading the API"""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = [self.get_embedding(text) for text in batch]
            # Filter out None values in case of failures
            batch_embeddings = [emb for emb in batch_embeddings if emb is not None]
            all_embeddings.extend(batch_embeddings)
            # Small delay to prevent overwhelming the API
            time.sleep(0.1)
        return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Process node and edge text data using Ollama embeddings and save as .pt files."
    )
    parser.add_argument(
        "--pkl_file",
        default="data_preprocess/Dataset/cora/processed/cora.pkl",
        type=str,
        help="Path to the Textual-Edge Graph .pkl file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3:8b",
        help="Name of the Ollama model",
    )
    parser.add_argument(
        "--name", type=str, default="cora", help="Prefix name for the output files"
    )
    parser.add_argument(
        "--path", type=str, default="data_preprocess/Dataset/cora/emb", help="Path to save the .pt files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of texts to process in one batch",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")

    # Parse arguments
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip("/"))
    Feature_path = os.path.join(base_dir, args.path)
    
    if not os.path.exists(Feature_path):
        os.makedirs(Feature_path)
    
    print(f"Embedding Model: {args.model_name}")
    
    output_file = os.path.join(
        Feature_path,
        args.name + "_" + args.model_name.replace(":", "_")
    )
    
    print("output_file:", output_file)
    
    # Load pickle file
    pkl_file = os.path.join(base_dir, args.pkl_file)
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    # Initialize the embedding generator
    generator = OllamaEmbeddingGenerator(
        model_name=args.model_name,
        batch_size=args.batch_size
    )
    
    # Get the embedding dimension
    hidden_size = generator.get_embedding_dimension()
    print(f"Embedding dimension: {hidden_size}")
    
    # Check if embeddings already exist
    if not os.path.exists(output_file + "_node.pt") or not os.path.exists(output_file + "_edge.pt"):
        # Process node text data
        print(f"Processing {len(data.text_nodes)} nodes...")
        node_embeddings_list = generator.get_batch_embeddings(data.text_nodes)
        
        if len(node_embeddings_list) > 0:
            node_embeddings = torch.tensor(node_embeddings_list)
            torch.save(node_embeddings, output_file + "_node.pt")
            print(f"Node embeddings saved to {output_file}_node.pt")
        else:
            print("Warning: No node embeddings were generated")
        
        # Process edge text data if available
        if len(data.text_edges) > 0:
            print(f"Processing {len(data.text_edges)} edges...")
            edge_embeddings_list = generator.get_batch_embeddings(data.text_edges)
            
            if len(edge_embeddings_list) > 0:
                edge_embeddings = torch.tensor(edge_embeddings_list)
                torch.save(edge_embeddings, output_file + "_edge.pt")
                print(f"Edge embeddings saved to {output_file}_edge.pt")
            else:
                print("Warning: No edge embeddings were generated")
        else:
            print("text_edges is empty, performing Xavier uniform initialization...")
            edge_embeddings = torch.empty((data.edge_index.shape[1], hidden_size))  # edges x dimension
            torch.nn.init.xavier_uniform_(edge_embeddings)
            torch.save(edge_embeddings, output_file + "_edge.pt")
            print(f"Randomly initialized edge embeddings saved to {output_file}_edge.pt")
    else:
        print("Existing saved embeddings found")


if __name__ == "__main__":
    main()

# python data_preprocess/generate_llama_bert.py --pkl_file data_preprocess/Dataset/amazon_apps/processed/apps.pkl --model_name llama3:8b --name apps --path data_preprocess/Dataset/amazon_apps/emb --batch_size 32
