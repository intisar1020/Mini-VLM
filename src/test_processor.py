from transformers import AutoTokenizer
from PIL import Image
import numpy as np

from processing_paligemma import PaliGemmaProcessor

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer)
num_image_tokens = 1
image_size = 224
processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
text_prompt = ["hello world"]
dummy_image_array = np.random.rand(100, 100, 3).astype(np.float32)
dummy_image = Image.fromarray(
    (dummy_image_array * 255).astype(np.uint8)
)  # Convert back to uint8 for PIL
images = [dummy_image]


data = processor(text=text_prompt, images=images)
for k, v in data.items():
    print(k, v.shape)

input_ids = data["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("\nToken ID -> Token Text:")
for token_id, token_text in zip(input_ids.tolist(), tokens):
    print(f"{token_id} -> '{token_text}'")
