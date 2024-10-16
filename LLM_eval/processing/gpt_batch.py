import json
import glob
from tqdm import tqdm
import base64
import jsonlines
import os
import boto3

{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

inp = "Describe the image in detail. Start with a high-level description: Begin by providing an overall description of the image, capturing its main subject or scene. Describe the visual elements: Break down the image into its key visual elements and describe them in detail, including color: start from left part of the picture and proceed to the right, mentioning what is placed next to what or under/on what. Finally, avoid fabricating information."


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
s3_client = boto3.client('s3')
bucket_name = 'egopics'
jpg_files_urls = []

def generate_presigned_url(bucket_name, key, expiration=172800):
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': key},
        ExpiresIn=expiration
    )
    return url

# List to hold the presigned URLs
jpg_files_urls = []

# List all objects and generate presigned URLs
def list_files_and_generate_urls(bucket_name, prefix=''):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.jpg'):
                    presigned_url = generate_presigned_url(bucket_name, obj['Key'])
                    jpg_files_urls.append(presigned_url)

# Start with an empty prefix to list all objects
#list_files_and_generate_urls(bucket_name)

ap = 0
unprocessed = []
file_name = "error.jsonl"
with jsonlines.open(file_name) as f:
    for line in f.iter():
        ap+=1
        presigned_url = generate_presigned_url(bucket_name, line["custom_id"])
        jpg_files_urls.append(presigned_url)
print(ap)


frames = []
# Print the presigned URLs of all .jpg files
for url_ in tqdm(jpg_files_urls, desc="Uploading"):
    #print(url) https://egopics.s3.ap-southeast-2.amazonaws.com/third/97b228e5-1817-4ca8-b596-ae1bb79ee6e2/frame465.jpg?AWSAccessKeyId=AKIAQ3EGVM7CVTWO2YND&Signature=VzgwRTvE%2BZ3Z4Woqlig9ZFPhgWQ%3D&Expires=1721096446
    new_name = url_.split("/")[-3]+"/"+url_.split("/")[-2]+"/"+url_.split("/")[-1].split("?")[0]


    task = {
        "custom_id": new_name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 100,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a smart assistant generating interior descriptions"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":inp},
                        
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url_
                            }
                        },
                    ],
                }
            ]            
        }
    }
    
    frames.append(task)
file_name = "batch_tasks.jsonl"
with open(file_name, 'w') as file:
    for obj in tqdm(frames, desc="Writing"):
        file.write(json.dumps(obj) + '\n')

# #data = json.load(open(file_name))
# # with jsonlines.open(file_name) as f:

# #     for line in f.iter():

# #         print(line["body"]["messages"][0]["content"]) 



