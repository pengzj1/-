import argparse
import pickle
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    PreTrainedModel,
    Trainer,
)
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

class MeanPoolingModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.model = model
        self.config = model.config

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return TokenClassifierOutput(logits=mean_embeddings)

def main():
    parser = argparse.ArgumentParser(description="Llama文本嵌入生成")
    parser.add_argument(
        "--pkl_file",
        default="twitter/processed/twitter.pkl",
        type=str,
        help="输入数据文件路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Llama模型名称或路径"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="twitter",
        help="输出文件前缀"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="twitter/emb",
        help="输出路径"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="最大序列长度"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="推理批大小"
    )
    parser.add_argument(
        "--use_flash",
        action="store_true",
        help="启用Flash Attention"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备"
    )
    parser.add_argument(
        "--truncation_side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="截断方向"
    )

    args = parser.parse_args()

    # 路径设置
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip("/"))
    output_dir = os.path.join(base_dir, args.path)
    cache_dir = os.path.join(output_dir, "cache")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 加载数据
    with open(os.path.join(base_dir, args.pkl_file), "rb") as f:
        data = pickle.load(f)
    all_texts = data.text_nodes + data.text_edges

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        truncation_side=args.truncation_side
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 自定义批处理函数
    def batch_tokenize(batch_texts):
        batch_encodings = []
        for text in batch_texts:
            encoding = tokenizer(
                text,
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True
            )
            batch_encodings.append(encoding["input_ids"].squeeze(0))
        
        padded = pad_sequence(
            batch_encodings,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        return {
            "input_ids": padded,
            "attention_mask": (padded != tokenizer.pad_token_id).long()
        }

    # 创建数据集
    dataset = Dataset.from_dict({"text": all_texts})
    dataset = dataset.map(
        lambda batch: batch_tokenize(batch["text"]),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["text"]
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="flash_attention_2" if args.use_flash else "eager",
        device_map="auto"
    )
    model.eval()

    # 初始化自定义模型
    pooling_model = MeanPoolingModel(model)

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir=cache_dir,
        per_device_eval_batch_size=args.batch_size,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        report_to="none",
        logging_steps=50,
        remove_unused_columns=False
    )

    # 初始化Trainer
    trainer = Trainer(
        model=pooling_model,
        args=training_args,
    )

    # 执行推理
    predictions = trainer.predict(dataset)
    embeddings = torch.tensor(predictions.predictions)

    # 分割并保存结果
    node_embeddings = embeddings[:len(data.text_nodes)]
    edge_embeddings = embeddings[len(data.text_nodes):]

    output_prefix = os.path.join(
        output_dir,
        f"{args.name}_llama_{args.max_length}"
    )

    torch.save(node_embeddings, f"{output_prefix}_node.pt")
    torch.save(edge_embeddings, f"{output_prefix}_edge.pt")

    print(f"嵌入已保存至: {output_prefix}_[node|edge].pt")

if __name__ == "__main__":
    main()

# python llama_embeddings.py \
  --model_name meta-llama/Llama-3-8B-Instruct \
  --max_length 4096 \
  --batch_size 8 \
  --use_flash \
  --truncation_side left
