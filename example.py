from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

model_path = "/home/zyz/qwen25vl/model/Qwen2___5-VL-3B-Instruct"
img_path = "/home/zyz/qwen25vl/test/微信图片_20250724092349.jpg"
question = "请用中文描述一下这张图片的内容。给出一些评价。"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

image = Image.open(img_path)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ],
}]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=2000)
ans = processor.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
print(ans)
