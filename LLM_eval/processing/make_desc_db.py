import chromadb
import torch
import glob
import argparse
from chromadb.utils import embedding_functions
import json
import torch
from transformers import BitsAndBytesConfig
import requests
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoConfig, LlamaConfig 
from openai import OpenAI
import time
from modelling import get_model_llava, get_MGM
from memory.LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from memory.LLaVA.llava.mm_utils import process_images, tokenizer_image_token
from transformers import TextStreamer
import copy
from tqdm import tqdm
import os
# from memory.MGM.mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from memory.MGM.mgm.mm_utils import process_images, tokenizer_image_token

"""
This is the code to create descriptions for the filtered frames.
"""

keys = json.load(open('memory/my_keys.json'))
clientGPT = OpenAI(api_key=keys["openAI"])
MODEL="gpt-4o"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process two arguments.")
    
    # Add arguments
    #? part should be first...sixth, as name of folders I place videos in.
    parser.add_argument('part', type=str, help='Which of the folders to process')
    #parser.add_argument('json_number', type=str, help='How to name json file')
    

    args = parser.parse_args()


    #? Choose model.  conv, roles, tokenizer, model, image_processor, context_len = get_model_llava()
    conv, roles, tokenizer, model, image_processor, context_len = get_MGM()

    #* Checked, exists!

    folder_path = "ego4d/data/frames/"+args.part + "/*"
    #folder_path_ = "ego4d/data/frames/*/*/*.jpg"
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=keys["openAI"],
                    model_name="text-embedding-ada-002"
                )
    
    f_path = args.part + "/*.jpg"
    video_folders = glob.glob(folder_path)
    #video_folders = glob.glob(f_path)
    inp = "Describe the image in detail. Start with a high-level description: Begin by providing an overall description of the image, capturing its main subject or scene. Describe the visual elements: Break down the image into its key visual elements and describe them in detail, including color: start from left part of the picture and proceed to the right, mentioning what is placed next to what or under/on what. Finally, avoid fabricating information."
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = "</s>"
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    max_new_tokens = 400
    prompt = f"USER: <image>\n{inp}\nASSISTANT:"
    # videos = ["ego4d/data/frames/first/2fadb1b6-dfd6-4380-98a7-dc895c223848", "ego4d/data/frames/second/9e7fa594-5774-4681-a034-50a5c83cff8f"]

    
    
    for folder in video_folders:
        print("=================")
        print(folder)
        client = chromadb.PersistentClient(path=folder)
        #client.delete_collection("gpt") after_test_llava_single/after_test_llava
        collection = client.create_collection(
            name="llava",
            embedding_function=openai_ef,
        )
        # #print(folder)
        frames = glob.glob(folder+"/*.jpg")
        time_api = []
        time_col = []
        time_image = []
        for frame in frames:
            try:

                time_s_process = time.time()

                image = load_image(frame)
                

                image_size = image.size
                # Similar operation in model_worker.py
                image_tensor = process_images([image], image_processor, model.config)
                time_e_process = time.time()
                print("Image processing:  ", time_e_process-time_s_process)
                time_image.append(time_e_process-time_s_process)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                time_s_api = time.time()
                #image_sizes=[image_size],
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        images_aux=None,
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=400,
                        streamer=None,
                        use_cache=True)

                outputs = tokenizer.decode(output_ids[0]).strip()
                time_e_api = time.time()
                print("API processing:  ", time_e_api-time_s_api)  
                time_api.append(time_e_api-time_s_api)
                time_s_collection = time.time()

                collection.add(
                ids=[frame],
                documents=[outputs] # A list of numpy arrays representing images
            )
                time_e_collection = time.time()
                print("Collection processing:  ", time_e_collection-time_s_collection)
                time_col.append(time_e_collection-time_s_collection)
                del outputs
                torch.cuda.empty_cache()
            except KeyboardInterrupt:
                print(sum(time_image) / len(time_image))
                print(sum(time_api) / len(time_api))
                print(sum(time_col) / len(time_col))
                os._exit(0)

if __name__=="__main__":
    main()
