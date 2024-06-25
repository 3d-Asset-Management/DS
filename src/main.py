from fastapi import FastAPI
from typing import List
from huggingface_hub import login
import subprocess
import os
from datetime import datetime
import re
from PIL import Image
import io

import requests

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch

from transformers import pipeline

from fastapi import File, UploadFile

from pydantic import BaseModel

import subprocess

class tripled_response_json(BaseModel):
    model_path:str
    image_path:str
    date_of_creation: str
    description:str
    category:str


app = FastAPI()

@app.post("/generate_3d/" , response_model=tripled_response_json)
async def generate_3d(img_file : UploadFile = File(...) , img_id = "test_img"):

    """
    takes in image generates 3d model and returns all the metadata including the filepath for 3d object
    """
    
    contents = await img_file.read()
    # Open the image using PIL (optional: for image processing)

    img = Image.open(io.BytesIO(contents))

    img.save(f"load/images/{img_id}.png")

    # login to hugging face

    login("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")

    # create image id
    id = img_id
    
    # save image in a particular location and generate filepath 
    img_filepath = f"./load/images/{id}.png"

    # run command to train and generate 3d model 
        # Define the command to be executed
    command = [
        "python",
        "launch.py",
        "--config", "configs/stable-zero123.yaml",
        "--train",
        "--gpu", "0",
        f"data.image_path={img_filepath}",
        "trainer.max_steps=100"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Command executed successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing command:\n{e.stderr}")

    # obtain the save dir of the model
    base_dir = "outputs/zero123-sai/"
    
    current_path = None
    pattern = rf'^\[64, 128, 256\]_{id}\.png@.*?$'
    
    for fn in os.listdir(base_dir):
    
        print(fn)
        match = re.match(pattern, fn)
    
        if match:
            current_path = os.path.join(base_dir , fn)
            print(current_path)
            
    save_dir = os.path.join(current_path , "save")

    # Define the command to be executed
    command = [
        "python",
        "launch.py",
        "--config", f"{save_dir}/../configs/parsed.yaml",
        "--export",
        "--gpu", "0",
        f"resume={save_dir}/../ckpts/last.ckpt",
        "system.exporter_type=mesh-exporter",
        "system.exporter.context_type=cuda",
        "system.geometry.isosurface_threshold=15.0"
    ]
    
    # Run the command
    try:
        result = subprocess.run(command)
        print(f"Command executed successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing command:\n{e.stderr}")

    final_path = os.path.join(base_dir , id)
    os.rename(current_path, final_path)

    # lvm 

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

    # tags = lvm(img) 

    prompt = "USER: <image>\nClassify this item as one of the following: food,beverage,cosmetics,dairy\nASSISTANT:"
    outputs = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    txt = outputs[0]["generated_text"]
    cat = txt.split("ASSISTANT: ", 1)[-1]

    prompt = "USER: <image>\nGenerate a description for the image. Include its physical desription, its uses, approximate cost, benefits and demerits.\nASSISTANT:"
    outputs = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    txt = outputs[0]["generated_text"]
    desc = txt.split("ASSISTANT: ", 1)[-1]

    # save 3d model in a specified location 

    model_path = os.path.join(final_path , "save/it100-export")
    os.rename(img_filepath , os.path.join(model_path , f"{id}.png"))

    s3_link = f"s3://shaurya-bucket-1234/{id}"

    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', 'model_path', s3_link, '--recursive'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("Upload Successful")
    except subprocess.CalledProcessError as e:
        print("STDERR:", e.stderr)
        print("Upload Failed")
    except FileNotFoundError:
        print("AWS CLI not found. Please ensure AWS CLI is installed and in your PATH.")

    # s3_link = save_3d_at_s3_bucket() 

    # prepare and return json that is to be sent
    return {
        "s3_link":s3_link,
        "date_of_creation": (datetime.now().strftime("%H:%M:%S , %d-%m-%Y,")),
        "description":desc,
        "category":cat
    }