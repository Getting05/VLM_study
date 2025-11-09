import os
import json
import numpy as np
import torch
from typing import Any, Dict, List
from PIL import Image
from datasets import Dataset, DatasetDict
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
)
import importlib
import matplotlib.pyplot as plt
from swanlab.integration.transformers import SwanLabCallback
from dotenv import load_dotenv

"""数据整理器类：将批次数据转换为模型输入格式"""
class Qwen3VLDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 处理文本输入和标签
        input_id_tensors = [torch.as_tensor(s["input_ids"], dtype=torch.long) for s in features]
        attention_tensors = [torch.as_tensor(s["attention_mask"], dtype=torch.long) for s in features]
        label_tensors = [torch.as_tensor(s["labels"], dtype=torch.long) for s in features]

        max_length = max(t.size(0) for t in input_id_tensors)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("需定义pad_token_id或eos_token_id")

        input_ids = torch.full((len(features), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)

        for idx, (ids, attn, lbl) in enumerate(zip(input_id_tensors, attention_tensors, label_tensors)):
            length = ids.size(0)
            input_ids[idx, :length] = ids
            attention_mask[idx, :length] = attn
            labels[idx, :length] = lbl

        # 处理图像数据
        pixel_tensors = []
        for s in features:
            pv = s["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_tensors.append(pv)
        pixel_values = torch.cat(pixel_tensors, dim=0)

        image_grid_thw = torch.stack(
            [torch.as_tensor(s["image_grid_thw"], dtype=torch.long).view(-1) for s in features], dim=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


"""数据集加载函数：读取JSON标注和图像"""
def load_sewer_json_dataset(json_path, images_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 类别ID到名称的映射
    category_map = {cat["id"]: cat["name"] for cat in data["categories"]}
    
    # 按image_id分组标注
    annotations_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append({
            "category": category_map[ann["category_id"]],
            "bbox": ann["bbox"]  # [x, y, width, height]
        })
    
    # 构建数据集列表
    dataset_list = []
    for img_info in data["images"]:
        img_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])
        annotations = annotations_by_image.get(img_id, [])
        dataset_list.append({
            "image_path": img_path,
            "annotations": annotations,
            "height": img_info["height"],
            "width": img_info["width"]
        })
    
    return Dataset.from_list(dataset_list)


"""数据预处理函数：转换为模型输入格式"""
def process_func(example, tokenizer, processor):
    MAX_LENGTH = 8192
    # 读取图像
    image = Image.open(example["image_path"]).convert("RGB")
    
    # 构建输出文本（损坏类型+边界框）
    output_lines = []
    for ann in example["annotations"]:
        damage_type = ann["category"]
        bbox = ann["bbox"]
        output_lines.append(f"{damage_type}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    output_content = "\n".join(output_lines)
    
    # 构建对话消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,
    )
    
    # 处理输入IDs和注意力掩码
    instruction_input_ids = inputs["input_ids"][0]
    instruction_attention_mask = inputs["attention_mask"][0]
    instruction_pixel_values = inputs["pixel_values"]
    instruction_image_grid_thw = inputs["image_grid_thw"][0]
    
    # 处理响应（标签部分）
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    response_input_ids = response["input_ids"]
    response_attention_mask = response.get("attention_mask", [1] * len(response_input_ids))
    
    # 确保响应以eos结束
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        if not response_input_ids or response_input_ids[-1] != eos_token_id:
            response_input_ids.append(eos_token_id)
            response_attention_mask.append(1)
    else:
        pad_token_id = tokenizer.pad_token_id or 0
        response_input_ids.append(pad_token_id)
        response_attention_mask.append(1)
    
    # 拼接输入和标签
    input_ids = instruction_input_ids + response_input_ids
    attention_mask = instruction_attention_mask + response_attention_mask
    labels = [-100] * len(instruction_input_ids) + response_input_ids
    
    # 截断过长序列
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": instruction_pixel_values,
        "image_grid_thw": instruction_image_grid_thw,
    }


"""评估指标计算：类型准确率和IoU"""
def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1, area2 = w1 * h1, w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    type_correct = 0
    iou_scores = []
    total_samples = len(decoded_preds)
    
    for pred, label in zip(decoded_preds, decoded_labels):
        # 解析真实标注
        label_regions = {}
        for line in label.strip().split("\n"):
            if ":" not in line:
                continue
            typ, bbox_str = line.split(":", 1)
            typ = typ.strip()
            try:
                bbox = eval(bbox_str.strip())
                if typ not in label_regions:
                    label_regions[typ] = []
                label_regions[typ].append(bbox)
            except:
                continue
        
        # 解析预测结果
        pred_regions = {}
        for line in pred.strip().split("\n"):
            if ":" not in line:
                continue
            typ, bbox_str = line.split(":", 1)
            typ = typ.strip()
            try:
                bbox = eval(bbox_str.strip())
                if typ not in pred_regions:
                    pred_regions[typ] = []
                pred_regions[typ].append(bbox)
            except:
                continue
        
        # 计算类型准确率
        true_types = set(label_regions.keys())
        pred_types = set(pred_regions.keys())
        if true_types:
            type_correct += len(true_types & pred_types) / len(true_types)
        
        # 计算IoU
        for typ, true_bboxes in label_regions.items():
            if typ not in pred_regions:
                continue
            pred_bboxes = pred_regions[typ]
            for true_bbox in true_bboxes:
                max_iou = max([compute_iou(true_bbox, p_bbox) for p_bbox in pred_bboxes])
                iou_scores.append(max_iou)
    
    return {
        "type_accuracy": type_correct / total_samples if total_samples > 0 else 0,
        "mean_iou": np.mean(iou_scores) if iou_scores else 0
    }


"""主训练函数"""
def main():
    # 配置参数
    global PROMPT_TEXT
    PROMPT_TEXT = """识别下水道图像中的所有损坏区域。每个区域输出：
- 损坏类型（如cr、jg）
- 边界框[x, y, width, height]（像素坐标）
格式："类型: [x, y, width, height]"（每行一个区域）"""
    
    # 环境变量和路径设置
    load_dotenv()
    os.environ["SWANLAB_API_KEY"] = os.getenv("SWAN_LAB", "")
    
    # ==== 请修改为你的数据集路径 ====
    train_json_path = "/home/zyz/xingrong/data/final_all_datasets/train_coco.json"    # 训练集JSON标注
    train_images_dir = "/home/zyz/xingrong/data/final_all_datasets/images/train"             # 训练集图像文件夹
    test_json_path = "/home/zyz/xingrong/data/final_all_datasets/val_coco.json"      # 测试集JSON标注
    test_images_dir = "/home/zyz/xingrong/data/final_all_datasets/images/val"               # 测试集图像文件夹
    model_id = "/home/zyz/qwen25vl/model/Qwen/Qwen2___5-VL-3B-Instruct"  # 模型路径
    output_dir = "/home/zyz/qwen25vl/ex1/sewer_result"          # 输出路径
    # ==============================
    
    # 加载数据集
    print("加载数据集...")
    train_data = load_sewer_json_dataset(train_json_path, train_images_dir)
    test_data = load_sewer_json_dataset(test_json_path, test_images_dir)
    train_data = train_data.shuffle(seed=222)
    test_data = test_data.shuffle(seed=222)
    print(f"训练数据大小: {len(train_data)}, 测试数据大小: {len(test_data)}")
    
    # 加载模型和处理器
    print("加载模型和处理器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True
    )
    
    # 加载模型配置和模型
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    module = importlib.import_module(f"transformers.models.{config.model_type}.modeling_{config.model_type}")
    model_cls = getattr(module, arch)
    model = model_cls.from_pretrained(
        model_id,
        cache_dir=os.environ.get("HF_HOME", "./"),
        device_map="auto",
        trust_remote_code=True,
    )
    model.to(dtype=torch.bfloat16)
    model.config.use_cache = False
    
    # 预处理数据集
    print("预处理数据集...")
    map_kwargs = {"tokenizer": tokenizer, "processor": processor}
    train_dataset = train_data.map(
        process_func,
        remove_columns=train_data.column_names,
        fn_kwargs=map_kwargs,
        num_proc=4  # 多进程加速
    )
    eval_dataset = test_data.map(
        process_func,
        remove_columns=test_data.column_names,
        fn_kwargs=map_kwargs,
        num_proc=4
    )
    lora_config_dict = {
        "lora_rank": 128, # 低秩矩阵的秩
        "lora_alpha": 16, # LoRA 的缩放因子
        "lora_dropout": 0, # LoRA 的 dropout 概率
    }
    # 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=target_modules,
        inference_mode=False,
        r=lora_config_dict["lora_rank"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias="none",
    )

    peft_model = get_peft_model(model, config)
    
    peft_model.enable_input_require_grads()

    print(f"可训练参数: {peft_model.print_trainable_parameters()}")
    
    # 配置训练参数
    swanlab_callback = SwanLabCallback(
        project="Qwen2.5-VL-Sewer-Inspection",
        experiment_name="sewer-damage-detection",
        config={
            "model": model_id,
            "dataset": "custom_sewer_dataset",
            "lora_rank": lora_config_dict["lora_rank"],
            "lora_alpha": lora_config_dict["lora_alpha"],
            "lora_dropout": lora_config_dict["lora_dropout"],
            "train_samples": len(train_data),
        },
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        logging_first_step=True,
        num_train_epochs=10,
        save_steps=50,
        save_total_limit=3,
        learning_rate=3e-5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="steps",
        eval_steps=50,
        report_to="none",
        fp16=True,  # 若GPU支持，启用混合精度训练
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Qwen3VLDataCollator(tokenizer=tokenizer),
        callbacks=[swanlab_callback],
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存结果
    print("保存模型和结果...")
    os.makedirs(output_dir, exist_ok=True)
    # 绘制损失曲线
    logs = trainer.state.log_history
    steps = [log['step'] for log in logs if 'loss' in log]
    losses = [log['loss'] for log in logs if 'loss' in log]
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    
    # 保存模型
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("训练完成!")


if __name__ == "__main__":
    main()
