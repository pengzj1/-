import argparse
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_embeddings(bert_path, llama_path):
    """
    Load embeddings from both BERT and Llama models, handling different naming conventions
    BERT: *_cls_node.pt, *_cls_edge.pt
    Llama: *_node.pt, *_edge.pt
    """
    # Load node embeddings
    bert_node_path = bert_path + "_cls_node.pt"
    llama_node_path = llama_path + "_node.pt"
    
    if not os.path.exists(bert_node_path):
        raise FileNotFoundError(f"BERT node embedding file not found: {bert_node_path}")
    if not os.path.exists(llama_node_path):
        raise FileNotFoundError(f"Llama node embedding file not found: {llama_node_path}")
    
    bert_node_emb = torch.load(bert_node_path)
    llama_node_emb = torch.load(llama_node_path)
    
    print(f"Loaded BERT node embeddings with shape: {bert_node_emb.shape}")
    print(f"Loaded Llama node embeddings with shape: {llama_node_emb.shape}")
    
    # Check for dimension mismatch in the first dimension (number of nodes)
    if bert_node_emb.shape[0] != llama_node_emb.shape[0]:
        raise ValueError(f"Node embedding count mismatch: BERT has {bert_node_emb.shape[0]} embeddings, Llama has {llama_node_emb.shape[0]}")
    
    # Load edge embeddings
    bert_edge_path = bert_path + "_cls_edge.pt"
    llama_edge_path = llama_path + "_edge.pt"
    
    if not os.path.exists(bert_edge_path):
        print(f"Warning: BERT edge embedding file not found: {bert_edge_path}")
        bert_edge_emb, llama_edge_emb = None, None
    elif not os.path.exists(llama_edge_path):
        print(f"Warning: Llama edge embedding file not found: {llama_edge_path}")
        bert_edge_emb, llama_edge_emb = None, None
    else:
        bert_edge_emb = torch.load(bert_edge_path)
        llama_edge_emb = torch.load(llama_edge_path)
        
        print(f"Loaded BERT edge embeddings with shape: {bert_edge_emb.shape}")
        print(f"Loaded Llama edge embeddings with shape: {llama_edge_emb.shape}")
        
        # Check for dimension mismatch in the first dimension (number of edges)
        if bert_edge_emb.shape[0] != llama_edge_emb.shape[0]:
            print(f"Warning: Edge embedding count mismatch: BERT has {bert_edge_emb.shape[0]} embeddings, Llama has {llama_edge_emb.shape[0]}")
            bert_edge_emb, llama_edge_emb = None, None
    
    return (bert_node_emb, llama_node_emb), (bert_edge_emb, llama_edge_emb)


def project_to_common_dimension(emb1, emb2, target_dim=None):
    """
    Project embeddings to a common dimensionality
    """
    # Get dimensions
    dim1 = emb1.shape[1]
    dim2 = emb2.shape[1]
    
    # Set target dimension if not provided
    if target_dim is None:
        target_dim = max(dim1, dim2)
    
    # Project embeddings to target dimension
    if dim1 != target_dim:
        # Use PCA to reduce dimension or zero padding to increase dimension
        if dim1 > target_dim:
            pca = PCA(n_components=target_dim)
            emb1_projected = torch.tensor(pca.fit_transform(emb1.detach().cpu().numpy())).to(emb1.device)
            print(f"Reduced emb1 dimension from {dim1} to {target_dim}")
        else:  # dim1 < target_dim
            padding = torch.zeros((emb1.shape[0], target_dim - dim1), device=emb1.device)
            emb1_projected = torch.cat([emb1, padding], dim=1)
            print(f"Increased emb1 dimension from {dim1} to {target_dim} with zero padding")
    else:
        emb1_projected = emb1
    
    if dim2 != target_dim:
        if dim2 > target_dim:
            pca = PCA(n_components=target_dim)
            emb2_projected = torch.tensor(pca.fit_transform(emb2.detach().cpu().numpy())).to(emb2.device)
            print(f"Reduced emb2 dimension from {dim2} to {target_dim}")
        else:  # dim2 < target_dim
            padding = torch.zeros((emb2.shape[0], target_dim - dim2), device=emb2.device)
            emb2_projected = torch.cat([emb2, padding], dim=1)
            print(f"Increased emb2 dimension from {dim2} to {target_dim} with zero padding")
    else:
        emb2_projected = emb2
    
    return emb1_projected, emb2_projected


def simple_fusion(bert_emb, llama_emb, weight=0.5, target_dim=None):
    """Combine embeddings with a weighted average after projecting to a common dimension"""
    # First ensure both embeddings have the same dimension
    if bert_emb.shape[1] != llama_emb.shape[1]:
        bert_emb_proj, llama_emb_proj = project_to_common_dimension(bert_emb, llama_emb, target_dim)
    else:
        bert_emb_proj, llama_emb_proj = bert_emb, llama_emb
    
    # Normalize embeddings to unit length
    bert_emb_norm = bert_emb_proj / bert_emb_proj.norm(dim=1, keepdim=True)
    llama_emb_norm = llama_emb_proj / llama_emb_proj.norm(dim=1, keepdim=True)
    
    # Weighted average
    fused_emb = weight * bert_emb_norm + (1 - weight) * llama_emb_norm
    
    # Normalize the fused embeddings
    fused_emb = fused_emb / fused_emb.norm(dim=1, keepdim=True)
    
    return fused_emb


def concat_fusion(bert_emb, llama_emb, target_dim=None):
    """Concatenate embeddings and optionally reduce dimension with PCA"""
    # Concatenate along feature dimension
    concat_emb = torch.cat([bert_emb, llama_emb], dim=1)
    
    # Apply PCA if target dimension is specified
    if target_dim is not None:
        pca = PCA(n_components=target_dim)
        concat_emb_np = concat_emb.cpu().numpy()
        reduced_emb = pca.fit_transform(concat_emb_np)
        
        # Convert back to torch tensor
        concat_emb = torch.from_numpy(reduced_emb).float()
        
        print(f"Reduced dimension from {bert_emb.shape[1] + llama_emb.shape[1]} to {target_dim}")
        print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
        
    return concat_emb


def attention_fusion(bert_emb, llama_emb, target_dim=None):
    """
    Fusion using learned attention weights, handling different embedding dimensions
    """
    # Get dimensions
    bert_dim = bert_emb.shape[1]
    llama_dim = llama_emb.shape[1]
    
    # Calculate dimension-based weights
    total_dim = bert_dim + llama_dim
    bert_weight = bert_dim / total_dim
    
    print(f"Model dimensions - BERT: {bert_dim}, Llama: {llama_dim}")
    print(f"Dimension-based weights - BERT: {bert_weight:.4f}, Llama: {1-bert_weight:.4f}")
    
    # Apply weighted fusion with projection
    return simple_fusion(bert_emb, llama_emb, weight=bert_weight, target_dim=target_dim)


def visualize_embeddings(embeddings_list, labels, output_path=None):
    """Visualize embeddings using PCA"""
    # Make sure all embeddings have the same dimensionality for fair comparison
    dims = [emb.shape[1] for emb in embeddings_list]
    
    if len(set(dims)) > 1:
        # Project to common dimension if they differ
        min_dim = min(dims)
        for i in range(len(embeddings_list)):
            if embeddings_list[i].shape[1] > min_dim:
                pca = PCA(n_components=min_dim)
                embeddings_list[i] = torch.tensor(
                    pca.fit_transform(embeddings_list[i].cpu().numpy())
                )
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for i, (embeddings, label) in enumerate(zip(embeddings_list, labels)):
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        reduced_emb = pca.fit_transform(embeddings.cpu().numpy())
        
        # Plot
        plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], alpha=0.5, label=label, color=colors[i])
    
    plt.legend()
    plt.title('PCA Visualization of Embeddings')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Fuse BERT and Llama embeddings for graph data")
    
    parser.add_argument("--bert_emb_path", type=str, required=True, 
                        help="Path to BERT embeddings file (without _cls_node.pt suffix)")
    parser.add_argument("--llama_emb_path", type=str, required=True, 
                        help="Path to Llama embeddings file (without _node.pt suffix)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save fused embeddings")
    parser.add_argument("--fusion_method", type=str, default="weighted", 
                        choices=["weighted", "concat", "attention"],
                        help="Method to fuse embeddings")
    parser.add_argument("--weight", type=float, default=0.5,
                        help="Weight for BERT embeddings in weighted fusion (between 0 and 1)")
    parser.add_argument("--target_dim", type=int, default=None,
                        help="Target dimension for the fused embeddings")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of embeddings")
    
    args = parser.parse_args()
    
    # Load both node and edge embeddings
    print("Loading embeddings...")
    (bert_node_emb, llama_node_emb), (bert_edge_emb, llama_edge_emb) = load_embeddings(
        args.bert_emb_path, args.llama_emb_path
    )
    
    # Process node embeddings
    print("\nProcessing node embeddings...")
    
    # Apply fusion method
    if args.fusion_method == "weighted":
        fused_node_emb = simple_fusion(bert_node_emb, llama_node_emb, args.weight, args.target_dim)
        print(f"Applied weighted fusion with BERT weight = {args.weight}")
    elif args.fusion_method == "concat":
        fused_node_emb = concat_fusion(bert_node_emb, llama_node_emb, args.target_dim)
        print("Applied concatenation fusion")
    elif args.fusion_method == "attention":
        fused_node_emb = attention_fusion(bert_node_emb, llama_node_emb, args.target_dim)
        print("Applied attention-based fusion")
    
    # Save fused node embeddings
    node_output_file = args.output_path + "_fused_node.pt"
    torch.save(fused_node_emb, node_output_file)
    print(f"Fused node embeddings saved to {node_output_file}")
    
    # Visualize node embeddings if requested
    if args.visualize:
        print("Generating node embedding visualization...")
        visualize_embeddings(
            [bert_node_emb, llama_node_emb, fused_node_emb],
            ["BERT", "Llama", "Fused"],
            args.output_path + "_node_viz.png"
        )
    
    # Process edge embeddings if available
    if bert_edge_emb is not None and llama_edge_emb is not None:
        print("\nProcessing edge embeddings...")
        
        # Apply same fusion method
        if args.fusion_method == "weighted":
            fused_edge_emb = simple_fusion(bert_edge_emb, llama_edge_emb, args.weight, args.target_dim)
        elif args.fusion_method == "concat":
            fused_edge_emb = concat_fusion(bert_edge_emb, llama_edge_emb, args.target_dim)
        elif args.fusion_method == "attention":
            fused_edge_emb = attention_fusion(bert_edge_emb, llama_edge_emb, args.target_dim)
        
        # Save fused edge embeddings
        edge_output_file = args.output_path + "_fused_edge.pt"
        torch.save(fused_edge_emb, edge_output_file)
        print(f"Fused edge embeddings saved to {edge_output_file}")
        
        # Visualize edge embeddings if requested
        if args.visualize:
            print("Generating edge embedding visualization...")
            visualize_embeddings(
                [bert_edge_emb, llama_edge_emb, fused_edge_emb],
                ["BERT", "Llama", "Fused"],
                args.output_path + "_edge_viz.png"
            )
    else:
        print("\nSkipping edge embedding fusion as embeddings are not available or have mismatched dimensions")


if __name__ == "__main__":
    main()

# python fusion_embedding.py \
#   --bert_emb_path "data_preprocess/Dataset/cora/emb/cora_bert_large_uncased_512" \
#   --llama_emb_path "data_preprocess/Dataset/cora/emb/cora_llama3_8b" \
#   --output_path "data_preprocess/Dataset/cora/emb/cora_fused" \
#   --fusion_method "attention" \
#   --visualize //4096

# # Weighted fusion with custom weight
# python fusion_embedding.py \
#   --bert_emb_path "..." \
#   --llama_emb_path "..." \
#   --output_path "..." \
#   --fusion_method "weighted" \
#   --weight 0.7

# # Concatenation with dimension reduction
# python fusion_embedding.py \
#   --bert_emb_path "..." \
#   --llama_emb_path "..." \
#   --output_path "..." \
#   --fusion_method "concat" \
#   --target_dim 1024(large) 768(base)