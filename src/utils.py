import subprocess
from pydantic import BaseModel
from typing import Optional

def upload_s3(model_path , s3_link):
        
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', model_path, s3_link, '--recursive'],
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

class simple_response(BaseModel):
    message:str
    bucket_name:str
    img_id:str

class recursive_response(BaseModel):
    status:str  # 0 response not ready , 1 response ready

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    sort_by_date: Optional[bool] = False