from fastapi import FastAPI , File , UploadFile , Form , BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
import os
from datetime import datetime
from PIL import Image
import io
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

import requests

from pydantic import BaseModel

from img import ImgOps 
from dddgen import threestudio_dddgen
from mdgen import llava_mdgen
from db_insert import PineconeOps
from utils import upload_s3 , simple_response , recursive_response

import asyncio

from pinecone import Pinecone
from utils import SearchRequest
from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/generate_3d" , response_model=simple_response)
async def generate_3d(background_tasks: BackgroundTasks , file : UploadFile = File(...) , img_id : str = Form(...)):

    os.chdir("/workspace/threestudio")
    contents = await file.read()

    img_ops = ImgOps(contents , img_id)

    s3_link = f"s3://model-store-capstone/{img_id}"

    background_tasks.add_task(generate_3d_bg , img_ops , s3_link)
    
    return {"message": "Jai Shree Ram",
           "bucket_name": "model-store-capstone",
            "img_id":img_id
           }

def generate_3d_bg(img_ops , s3_link):

    """
    takes in image generates 3d model and returns all the metadata including the filepath for 3d object
    """

    # login to hugging face
    login("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")
    
    # run command to train and generate 3d model 
    #     Define the command to be executed
    
    generator = threestudio_dddgen(img_ops.img_filepath)
    generator.train()
    save_dir , current_path = generator.savedir("outputs/zero123-sai/" , rf'^\[64, 128, 256\]_{img_ops.img_id}\.png@.*?$')
    generator.extract_mesh(save_dir)
    final_path = generator.renamedir(current_path , "outputs/zero123-sai/" , img_ops.img_id)
    print("rename done")
    print("starting md generation")
 
    mdgen = llava_mdgen()

    prompt = "Classify this item as one of the following: food,beverage,cosmetics,dairy"
    cat = mdgen.get_metadata(img_ops.img , prompt)

    prompt = "Generate a description for the image. Include its physical desription, its uses, approximate cost, benefits and demerits."
    desc = mdgen.get_metadata(img_ops.img , prompt)

    print(cat , desc)

    # save 3d model in a specified location
    model_path = os.path.join(final_path , "save/it100-export")
    os.rename(img_ops.img_filepath , os.path.join(model_path , "image.png"))

    print("s3ops starting")
    s3_link = f"s3://model-store-capstone/{img_ops.img_id}"
    upload_s3(model_path , s3_link)
    print("s3 done")

    res =  {
        "bucket-name":"model-store-capstone",
        "img-id":img_ops.img_id,
        "date-of-creation": (datetime.now().strftime("%H:%M:%S , %d-%m-%Y")),
        "description":desc,
        "sub-category":cat
    }

    # Add to database
    print("db ops starting")
    database = PineconeOps()
    database.insert(res)

@app.post("/frontend_recursive" , response_model=recursive_response)
async def frontend_recursive(bucket_name : str = Form(...) , img_id : str = Form(...)):

    s3 = boto3.client('s3')
    
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=img_id, Delimiter='/')
    except (NoCredentialsError, PartialCredentialsError):
        print("AWS credentials not found or incomplete.")
        return {"status":"0"}
    
    if 'Contents' in response or 'CommonPrefixes' in response:
        return {"status":"1"}
        
    return {"status":"0"}


# findbyid_and_return_data
@app.get("/findbyid/{id}")
async def findbyid_and_return_data(id: str):

    pc = PineconeOps()
    index = pc.index

    result = index.query(
        namespace="ns1",
        id=id,
        top_k=1,
        include_values=False,
        include_metadata=True
    )
    metadata_list = []
    for match in result['matches']:
        metadata_list.append(match['metadata'])
    return metadata_list


# ......filterSearch......
@app.post("/filtersearch")
async def search_similar(search_request: SearchRequest):
    # Generate vector embedding for the search query

    pc = PineconeOps()
    query_vector = pc.generate_embedding(search_request.query)

    # Create filter for category and subcategory if provided
    filter_query = {}
    if search_request.category:
        filter_query["category"] = {"$eq": search_request.category}
    if search_request.subcategory:
        filter_query["subcategory"] = {"$eq": search_request.subcategory}
    # Query Pinecone for similar vectors with filtering and top_k=4
    result = pc.index.query(
        namespace="ns1",
        vector=query_vector,
        top_k=4,
        include_values=False,
        include_metadata=True,
        filter=filter_query
    )
    # Check if there are any matches
    if not result.matches:
        raise HTTPException(status_code=404, detail="No similar entries found")
    # Extract metadata from the matches
    metadata_list = [match['metadata'] for match in result.matches]
    # Sort matches by date_of_creation if requested
    if search_request.sort_by_date:
        metadata_list = sorted(metadata_list, key=lambda x: x.get('date_of_creation', ''), reverse=True)
    # Return the sorted metadata list
    return metadata_list


# get all 3dobject
@app.get("/getallitems")
async def get_all_items():

    pc = PineconeOps()
    results=[]
    for id in pc.index.list(namespace='ns1'):
        for i in id:
            # print(i)
            result = pc.index.query(
                namespace="ns1",
                id=str(i),
                top_k=1,
                include_values=False,
                include_metadata=True
            )
            metadata_list = []
            for match in result['matches']:
                metadata_list.append(match['metadata'])
            # print(metadata_list)
            results.extend(metadata_list)
    return results


#  search endpoint
@app.get("/search")
async def search_similar(query: str):

    pc = PineconeOps()
    query_vector = pc.generate_embedding(query)
    result = pc.index.query(
        namespace="ns1",
        vector=query_vector,
        top_k=1,
        include_values=True,
        include_metadata=True
    )
    metadata_list = []
    for match in result['matches']:
        metadata_list.append(match['metadata'])
    return metadata_list
