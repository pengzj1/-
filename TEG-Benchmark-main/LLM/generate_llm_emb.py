import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
from sklearn.decomposition import PCA
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    PreTrainedModel,
    Trainer,
)
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset


class CLSEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.encoder = model

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        # Extract outputs from the model3
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        node_cls_emb = outputs.last_hidden_state[:, 0, :]  # Last layer
        return TokenClassifierOutput(logits=node_cls_emb)



def main():
    parser = argparse.ArgumentParser(
        description="Process node and edge text data and save the overall representation as .pt files."
    )
    parser.add_argument(
        "--pkl_file",
        default="twitter/processed/twitter.pkl",
        type=str,
        help="Path to the Textual-Edge Graph .pkl file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name or path of the Huggingface model",
    )
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--name", type=str, default="twitter", help="Prefix name for the  NPY file"
    )
    parser.add_argument(
        "--path", type=str, default="twitter/emb", help="Path to the .pt File"
    )
    parser.add_argument(
        "--pretrain_path", type=str, default=None, help="Path to the NPY File"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of the text for language models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Number of batch size for inference",
    )
    parser.add_argument("--fp16", type=bool, default=True, help="if fp16")
    parser.add_argument(
        "--cls",
        action="store_true",
        default=True,
        help="whether use cls token to represent the whole text",
    )
    parser.add_argument(
        "--nomask",
        action="store_true",
        help="whether do not use mask to claculate the mean pooling",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")


    parser.add_argument("--norm", type=bool, default=False, help="nomic use True")

    # 解析命令行参数
    args = parser.parse_args()
    pkl_file = args.pkl_file
    model_name = args.model_name
    name = args.name
    max_length = args.max_length
    batch_size = args.batch_size

    tokenizer_name = args.tokenizer_name

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip("/"))
    Feature_path = os.path.join(base_dir, args.path)
    cache_path = f"{Feature_path}cache/"

    if not os.path.exists(Feature_path):
        os.makedirs(Feature_path)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    print("Embedding Model:", model_name)

    if args.pretrain_path is not None:
        output_file = os.path.join(
            Feature_path
            , name
            + "_"
            + model_name.split("/")[-1].replace("-", "_")
            + "_"
            + str(max_length)
            + "_"
            + "Tuned"
        )
    else:
        output_file = os.path.join(
            Feature_path
            , name
            + "_"
            + model_name.split("/")[-1].replace("-", "_")
            + "_"
            + str(max_length)
        )

    print("output_file:", output_file)

    # ...existing code...

    pkl_file = os.path.join(base_dir, args.pkl_file)
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    # Check if text_edges exists and is not empty
    has_edge_text = hasattr(data, 'text_edges') and len(data.text_edges) > 0
    
    # If has edge text, combine with node text. Otherwise, just use node text
    if has_edge_text:
        text_data = data.text_nodes + data.text_edges
        print(f"Processing {len(data.text_nodes)} nodes and {len(data.text_edges)} edges...")
    else:
        text_data = data.text_nodes
        print(f"Processing {len(data.text_nodes)} nodes. No edge text found, will initialize randomly.")

    # Load tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = Dataset.from_dict({"text": text_data})  

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    dataset = dataset.map(tokenize_function, batched=True, batch_size=batch_size)

    if args.pretrain_path is not None:
        model = AutoModel.from_pretrained(f"{args.pretrain_path}",  attn_implementation="eager")
        print("Loading model from the path: {}".format(args.pretrain_path))
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")

    model = model.to(args.device)
    CLS_Feateres_Extractor = CLSEmbInfModel(model)
    CLS_Feateres_Extractor.eval()

    inference_args = TrainingArguments(
        output_dir=cache_path,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=12,
        fp16_full_eval=args.fp16,
    )

    # CLS representatoin
    if args.cls:
        if not os.path.exists(output_file + "_cls_node.pt") or (has_edge_text and not os.path.exists(output_file + "_cls_edge.pt")) or (not has_edge_text and not os.path.exists(output_file + "_cls_edge.pt")):
            trainer = Trainer(model=CLS_Feateres_Extractor, args=inference_args)
            cls_emb = trainer.predict(dataset)
            
            # Always save node embeddings
            node_cls_emb = torch.from_numpy(cls_emb.predictions[: len(data.text_nodes)])
            torch.save(node_cls_emb, output_file + "_cls_node.pt")
            
            # Handle edge embeddings based on availability
            if has_edge_text:
                edge_cls_emb = torch.from_numpy(cls_emb.predictions[len(data.text_nodes):])
                torch.save(edge_cls_emb, output_file + "_cls_edge.pt")
                print(f"Node and edge embeddings saved to {output_file}_cls_node.pt and {output_file}_cls_edge.pt")
            else:
                # For missing edge text, initialize randomly with Xavier uniform
                # Get the embedding dimension from the node embeddings
                hidden_size = node_cls_emb.shape[1]
                
                # Check if data.edge_index exists
                if hasattr(data, 'edge_index'):
                    num_edges = data.edge_index.shape[1]
                else:
                    # Fallback if edge_index doesn't exist
                    print("Warning: edge_index not found, assuming 0 edges")
                    num_edges = 0
                
                edge_cls_emb = torch.empty((num_edges, hidden_size))
                torch.nn.init.xavier_uniform_(edge_cls_emb)
                torch.save(edge_cls_emb, output_file + "_cls_edge.pt")
                print(f"Node embeddings saved to {output_file}_cls_node.pt")
                print(f"Randomly initialized edge embeddings saved to {output_file}_cls_edge.pt")
        else:
            print("Existing saved CLS embeddings found")
    else:
        raise ValueError("others are not defined")


if __name__ == "__main__":
    main()
