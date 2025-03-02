python subgraph_classifier.py \
    --use_PLM_node data/CSTAG/Photo/Feature/children_gpt_node.pt \
    --use_PLM_edge data/CSTAG/Photo/Feature/children_gpt_edge.pt \
    --graph_path data/CSTAG/Photo/children.pkl \
    --gnn_model GAT \
    --hidden_channels 256 \
    --num_layers 3 \
    --num_subgraphs 2000 \
    --min_subgraph_size 20 \
    --epochs 100
