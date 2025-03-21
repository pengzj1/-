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
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        node_cls_emb = outputs.last_hidden_state[:, 0, :]  # Last layer
        return TokenClassifierOutput(logits=node_cls_emb)


def main():
    parser = argparse.ArgumentParser(
        description="Process text label data and save the representations as .pt files."
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
        default="bert-large-uncased",
        help="Name or path of the Huggingface model",
    )
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--name", type=str, default="cora", help="Prefix name for the  NPY file"
    )
    parser.add_argument(
        "--path", type=str, default="data_preprocess/Dataset/cora/emb", help="Path to the .pt File"
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
        default=250,
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
    cache_path = f"{Feature_path}/cache/"

    if not os.path.exists(Feature_path):
        os.makedirs(Feature_path)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    print("Embedding Model:", model_name)

    pkl_file = os.path.join(base_dir, args.pkl_file)
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # 确保text_node_labels字段存在
    if not hasattr(data, 'text_node_labels') and not isinstance(data, dict):
        raise ValueError("数据对象中找不到'text_node_labels'字段")
    
    # 获取标签文本
    if hasattr(data, 'text_node_labels'):
        label_texts = data.text_node_labels
    elif isinstance(data, dict) and 'text_node_labels' in data:
        label_texts = data['text_node_labels']
    else:
        raise ValueError("未能在数据对象中找到'text_node_labels'字段")
    
    print(f"提取到 {len(label_texts)} 个文本标签")
    
    # 转换为列表（如果不是）
    if not isinstance(label_texts, list):
        label_texts = list(label_texts)
    
    # 加载模型
    model = AutoModel.from_pretrained(model_name)
    hidden_size = model.config.hidden_size  # 获取模型的隐藏维度

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建标签数据集
    dataset = Dataset.from_dict({"text": label_texts})  

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    dataset = dataset.map(tokenize_function, batched=True, batch_size=batch_size)

    # 加载或初始化模型
    if args.pretrain_path is not None:
        model = AutoModel.from_pretrained(f"{args.pretrain_path}", attn_implementation="eager")
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

    # 定义保存路径
    labels_path = os.path.join(Feature_path, "labels.pt")
    
    # 如果文件不存在，则生成嵌入
    if not os.path.exists(labels_path):
        # 提取标签嵌入
        trainer = Trainer(model=CLS_Feateres_Extractor, args=inference_args)
        label_emb_output = trainer.predict(dataset)
        label_embeddings = torch.from_numpy(label_emb_output.predictions)
        
        # 保存标签嵌入
        torch.save(label_embeddings, labels_path)
        print(f"标签嵌入已保存至: {labels_path}, 形状: {label_embeddings.shape}")
        
        # 另外，将原始标签转换为数字标签并保存
        unique_labels = list(set(label_texts))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = torch.tensor([label_to_id[label] for label in label_texts])
        
        numeric_labels_path = os.path.join(Feature_path, "numeric_labels.pt")
        torch.save(numeric_labels, numeric_labels_path)
        print(f"数字标签保存成功! 形状: {numeric_labels.shape}")
        
    else:
        print(f"标签嵌入文件已存在: {labels_path}")
        label_embeddings = torch.load(labels_path)
        print(f"已加载标签嵌入，形状: {label_embeddings.shape}")


if __name__ == "__main__":
    main()