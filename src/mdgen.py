from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
from transformers import pipeline

class llava_mdgen():
    def __init__(self):

        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16)

        model_id = "llava-hf/llava-1.5-7b-hf"
        self.pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

    def get_md(self , img , prompt):

        prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        outputs = self.pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        txt = outputs[0]["generated_text"]
        res = txt.split("ASSISTANT: ", 1)[-1]

        return res

