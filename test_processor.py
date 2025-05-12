from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import torch

from processing_paligemma import PaliGemmaProcessor, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print (tokenizer)
num_image_tokens = 2
image_size = 224
processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
text_prompt = ['world world']
dummy_image_array = np.random.rand(100, 100, 3).astype(np.float32)
dummy_image = Image.fromarray((dummy_image_array * 255).astype(np.uint8)) # Convert back to uint8 for PIL
images = [dummy_image]


data = processor(text=text_prompt, images=images)
for k, v in data.items():
    print (k, v.shape)
print (data["input_ids"])