from pinecone import Pinecone , ServerlessSpec
from transformers import AutoModel, AutoTokenizer
import os
import torch
import uuid

class PineconeOps:
    def __init__(self , index_name="3rdasset"):
        
        #pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pinecone_api_key = "30ce88eb-c15b-4c07-83d9-f22f9e2ba959"
        self.pc = Pinecone(api_key=pinecone_api_key, environment="us-east-1")
        
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(index_name)

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)
        
    def generate_embedding(self , text:str):

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def insert(self , entry):

        vector = self.generate_embedding(entry["description"])
        unique_id = str(uuid.uuid4())

        metadata = {
            "unique_id" : unique_id,
            "bucket_name" : entry["bucket-name"],
            "img_id" : entry["img-id"],
            "category" : "fmcg",
            "subcategory" : entry["sub-category"],
            "description" : entry["description"],
            "date":entry["date-of-creation"]
        }
        vector_tuple = (unique_id, vector, metadata)

        self.index.upsert(
            vectors=[vector_tuple],
            namespace="ns1"
        )

        self.index.describe_index_stats()

        return {"message": "Entry added successfully"}

