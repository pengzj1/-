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
    AutoModelForCausalLM,  # Changed from AutoModel
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
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        # For LLaMA, we'll use the last hidden state of the first token as the embedding
        node_cls_emb = outputs.last_hidden_state[:, 0, :]  # Last layer, first token
        return TokenClassifierOutput(logits=node_cls_emb)


def main():
    parser = argparse.ArgumentParser(
        description="Process node and edge text data and save the overall representation as .pt files using LLaMA."
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
        default="meta-llama/Llama-2-7b-hf",  # Default to Llama-2-7b
        help="Name or path of the Huggingface LLaMA model",
    )
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name (defaults to model_name if not specified)")
    parser.add_argument(
        "--name", type=str, default="twitter", help="Prefix name for the NPY file"
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
        help="whether do not use mask to calculate the mean pooling",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")
    parser.add_argument("--norm", type=bool, default=False, help="nomic use True")

    # Parse command line arguments
    args = parser.parse_args()
    
    # Use model_name for tokenizer if not specified
    tokenizer_name = args.tokenizer_name or args.model_name

    pkl_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.pkl_file)
    
    # Load data
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    text_data = data.text_nodes + data.text_edges  

    # Load tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, trust_remote_code=True
    )

    # Add pad token if not existing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    dataset = Dataset.from_dict({"text": text_data})  

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")

    dataset = dataset.map(tokenize_function, batched=True, batch_size=args.batch_size)

    # Load model
    if args.pretrain_path is not None:
        model = AutoModelForCausalLM.from_pretrained(f"{args.pretrain_path}", attn_implementation="eager")
        print(f"Loading model from the path: {args.pretrain_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, attn_implementation="eager")

    model = model.to(args.device)
    CLS_Feateres_Extractor = CLSEmbInfModel(model)
    CLS_Feateres_Extractor.eval()

    # Prepare inference arguments
    inference_args = TrainingArguments(
        output_dir=f"{args.path}cache/",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=12,
        fp16_full_eval=args.fp16,
    )

    # Prepare output file path
    output_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.path,
        f"{args.name}_{args.model_name.split('/')[-1].replace('-', '_')}_{args.max_length}"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # CLS representation
    if args.cls:
        if not os.path.exists(output_file + "_cls_node.pt") or not os.path.exists(output_file + "_cls_edge.pt"):
            trainer = Trainer(model=CLS_Feateres_Extractor, args=inference_args)
            cls_emb = trainer.predict(dataset)
            node_cls_emb = torch.from_numpy(cls_emb.predictions[: len(data.text_nodes)])    
            edge_cls_emb = torch.from_numpy(cls_emb.predictions[len(data.text_nodes) :])
            torch.save(node_cls_emb, output_file + "_cls_node.pt")
            torch.save(edge_cls_emb, output_file + "_cls_edge.pt")
            print(f"Embeddings saved to {output_file}")
        else:
            print("Existing CLS embeddings found")
    else:
        raise ValueError("Other embedding methods are not defined")


if __name__ == "__main__":
    main()
    
#python script.py --model_name meta-llama/Llama-2-7b-hf --pkl_file path/to/your/file.pkl
