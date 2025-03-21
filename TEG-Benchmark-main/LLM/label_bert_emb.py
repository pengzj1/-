import pickle
import torch
import os
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

def load_data_from_pickle(pickle_path):
    """从pickle文件加载数据"""
    print(f"正在加载数据: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def generate_text_embeddings(texts, batch_size=32):
    """使用BERT为文本生成嵌入"""
    # 加载BERT模型和分词器
    print("加载BERT模型...")
    tokenizer = BertTokenizer.from_pretrained('/home/sysu/Documents/TEG-Benchmark-main/data_preprocess/bert_base_uncased')
    model = BertModel.from_pretrained('/home/sysu/Documents/TEG-Benchmark-main/data_preprocess/bert_base_uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    embeddings = []
    
    # 分批处理文本
    print("生成文本嵌入...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # 确保batch_texts中的每个项都是字符串
        batch_texts = [str(text) for text in batch_texts]
        
        # 对文本进行编码
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        ).to(device)
        
        # 生成嵌入
        with torch.no_grad():
            output = model(**encoded_input)
            
        # 使用[CLS]标记的最后隐藏状态作为句子表示
        # 维度: (batch_size, hidden_size)
        batch_embeddings = output.last_hidden_state[:, 0, :]
        embeddings.append(batch_embeddings.cpu())
    
    # 将所有嵌入连接成一个张量
    all_embeddings = torch.cat(embeddings, dim=0)
    return all_embeddings

def extract_and_save_embeddings(pickle_path, save_path):
    """从PKL文件提取标签，生成嵌入并保存"""
    # 加载数据
    data = load_data_from_pickle(pickle_path)
    
    # 提取文本标签
    if hasattr(data, 'text_node_labels'):
        text_labels = data.text_node_labels
    elif isinstance(data, dict) and 'text_node_labels' in data:
        text_labels = data['text_node_labels']
    else:
        raise ValueError("未能在数据对象中找到'text_node_labels'字段")
    
    print(f"提取到 {len(text_labels)} 个文本标签")
    
    # 转换为列表（如果不是）
    if not isinstance(text_labels, list):
        text_labels = list(text_labels)
    
    # 打印数据类型以进行调试
    print(f"标签类型: {type(text_labels)}")
    if len(text_labels) > 0:
        print(f"第一个标签类型: {type(text_labels[0])}")
        print(f"第一个标签示例: {text_labels[0]}")
    
    # 生成文本嵌入
    text_embeddings = generate_text_embeddings(text_labels)
    
    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)
    
    # 构建完整的保存路径
    labels_path = os.path.join(save_path, "labels.pt")
    numeric_labels_path = os.path.join(save_path, "numeric_labels.pt")
    
    # 保存文本嵌入
    print(f"保存嵌入到 {labels_path}")
    torch.save(text_embeddings, labels_path)
    print(f"嵌入保存成功! 形状: {text_embeddings.shape}")
    
    # 另外，将原始标签转换为数字标签并保存
    # 这一步是为了兼容多模态对齐模型的训练需求
    unique_labels = list(set([str(label) for label in text_labels]))  # 确保是字符串
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = torch.tensor([label_to_id[str(label)] for label in text_labels])
    torch.save(numeric_labels, numeric_labels_path)
    print(f"数字标签保存成功! 形状: {numeric_labels.shape}")
    
    return text_embeddings, numeric_labels

if __name__ == "__main__":
    # 请替换为您的PKL文件路径和保存路径
    pickle_path = "data_preprocess/Dataset/amazon_apps/processed/apps.pkl"  # 修改为实际的pkl文件路径
    save_path = "data_preprocess/Dataset/amazon_apps/emb"  # 修改为实际的保存目录
    
    # 提取并保存嵌入
    text_embeddings, numeric_labels = extract_and_save_embeddings(pickle_path, save_path)
    
    # 显示一些统计信息
    print(f"\n生成的嵌入信息:")
    print(f"- 嵌入数量: {len(text_embeddings)}")
    print(f"- 嵌入维度: {text_embeddings.shape[1]}")
    print(f"- 标签类别数: {len(torch.unique(numeric_labels))}")