import torch
import os
import argparse
import numpy as np

def load_embeddings(bert_node_path, bert_edge_path, llama_node_path, llama_edge_path):
    """
    Load embeddings from the specified paths
    
    Args:
    - bert_node_path (str): Path to BERT node embeddings
    - bert_edge_path (str): Path to BERT edge embeddings
    - llama_node_path (str): Path to LLaMA node embeddings
    - llama_edge_path (str): Path to LLaMA edge embeddings
    
    Returns:
    - Tuple of (node_embeddings, edge_embeddings)
    """
    # Load BERT embeddings
    bert_node_emb = torch.load(bert_node_path)
    bert_edge_emb = torch.load(bert_edge_path)
    
    # Load LLaMA embeddings
    llama_node_emb = torch.load(llama_node_path)
    llama_edge_emb = torch.load(llama_edge_path)
    
    return (bert_node_emb, bert_edge_emb, llama_node_emb, llama_edge_emb)

def fuse_embeddings(bert_emb, llama_emb, method='concat', weight_bert=0.5):
    """
    Fuse embeddings using different methods
    
    Args:
    - bert_emb (torch.Tensor): BERT embeddings
    - llama_emb (torch.Tensor): LLaMA embeddings
    - method (str): Fusion method ('concat', 'weighted_sum', 'max_pooling')
    - weight_bert (float): Weight for BERT embeddings in weighted sum
    
    Returns:
    - Fused embeddings (torch.Tensor)
    """
    if method == 'concat':
        # Concatenate embeddings along the last dimension
        return torch.cat([bert_emb, llama_emb], dim=-1)
    
    elif method == 'weighted_sum':
        # Weighted sum of embeddings
        return weight_bert * bert_emb + (1 - weight_bert) * llama_emb
    
    elif method == 'max_pooling':
        # Element-wise max pooling
        return torch.max(bert_emb, llama_emb)
    
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fuse text embeddings from BERT and LLaMA")
    
    # BERT embedding paths
    parser.add_argument("--bert_node_emb", 
                        type=str, 
                        required=True, 
                        help="Path to BERT node embeddings")
    parser.add_argument("--bert_edge_emb", 
                        type=str, 
                        required=True, 
                        help="Path to BERT edge embeddings")
    
    # LLaMA embedding paths
    parser.add_argument("--llama_node_emb", 
                        type=str, 
                        required=True, 
                        help="Path to LLaMA node embeddings")
    parser.add_argument("--llama_edge_emb", 
                        type=str, 
                        required=True, 
                        help="Path to LLaMA edge embeddings")
    
    # Fusion method and parameters
    parser.add_argument("--fusion_method", 
                        type=str, 
                        choices=['concat', 'weighted_sum', 'max_pooling'], 
                        default='concat', 
                        help="Method to fuse embeddings")
    parser.add_argument("--bert_weight", 
                        type=float, 
                        default=0.5, 
                        help="Weight for BERT embeddings in weighted sum method")
    
    # Output paths
    parser.add_argument("--output_node_emb", 
                        type=str, 
                        default="fused_node_embeddings.pt", 
                        help="Output path for fused node embeddings")
    parser.add_argument("--output_edge_emb", 
                        type=str, 
                        default="fused_edge_embeddings.pt", 
                        help="Output path for fused edge embeddings")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load embeddings
    bert_node_emb, bert_edge_emb, llama_node_emb, llama_edge_emb = load_embeddings(
        args.bert_node_emb, 
        args.bert_edge_emb, 
        args.llama_node_emb, 
        args.llama_edge_emb
    )
    
    # Fuse node embeddings
    fused_node_emb = fuse_embeddings(
        bert_node_emb, 
        llama_node_emb, 
        method=args.fusion_method, 
        weight_bert=args.bert_weight
    )
    
    # Fuse edge embeddings
    fused_edge_emb = fuse_embeddings(
        bert_edge_emb, 
        llama_edge_emb, 
        method=args.fusion_method, 
        weight_bert=args.bert_weight
    )
    
    # Save fused embeddings
    torch.save(fused_node_emb, args.output_node_emb)
    torch.save(fused_edge_emb, args.output_edge_emb)
    
    print(f"Fused node embeddings saved to: {args.output_node_emb}")
    print(f"Fused edge embeddings saved to: {args.output_edge_emb}")

if __name__ == "__main__":
    main()

# python embedding_fusion.py \
#     --bert_node_emb bert_node_embeddings.pt \
#     --bert_edge_emb bert_edge_embeddings.pt \
#     --llama_node_emb llama_node_embeddings.pt \
#     --llama_edge_emb llama_edge_embeddings.pt \
#     --fusion_method concat \
#     --bert_weight 0.5 \
#     --output_node_emb fused_node_embeddings.pt \
#     --output_edge_emb fused_edge_embeddings.pt
