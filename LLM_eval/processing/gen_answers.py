import argparse
import json
import glob
import chromadb
import torch
import glob
import argparse
from chromadb.utils import embedding_functions
from openai import OpenAI
import torch
from transformers import BitsAndBytesConfig
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoConfig, LlamaConfig 
from modelling import get_MGM #get_model_llava  #, 
from memory.LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from memory.LLaVA.llava.mm_utils import process_images, tokenizer_image_token
from transformers import TextStreamer
import copy
from tqdm import tqdm
import time
import base64
# from memory.MGM.mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from memory.MGM.mgm.mm_utils import process_images, tokenizer_image_token
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import os


"""
This code:
1) Retrieves corresponding images and generates descriptions for them
 for LLaVA, Mini-Gemini. For GPT-4o see gpt_batch.py.
"""

questions = "ego4d/qaego4d/annotations.test.json"
#img_answers = "clip_mgm_whole.json"


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    


def main():
    """
    Finding answers in DB.
    1) For each question extract frame from corresponding video DB;
    2) For this image generate textual answer;
    4) Save in dict --> json file: sample_id: {video_id,  selected frame, answer}
    """
    whole = {}

    keys = json.load(open('my_keys.json'))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=keys["openAI"],
                    model_name="text-embedding-ada-002"
               )
    
    parser = argparse.ArgumentParser(description="Process two arguments.")
    #? part should be first...sixth, as name of folders I place 'em videos in.
    parser.add_argument('part', type=str, help='Which of the folders to process')
    args = parser.parse_args()    
    print(args.part)
    my_file = f"gpt_answers_img_only_{args.part}.json"
    folder_files = glob.glob("ego4d/data/frames/" + args.part + "/*")
    #folder_files = glob.glob("ego4d/data/frames/*/*")
    #print(folder_files)
    video_files = [s.split("/")[-1] for s in folder_files]
    real_files = json.load(open(questions))
    pic_answers = json.load(open("clip_mgm_whole.json"))
    clientGPT = OpenAI(api_key=keys["openAI"])
    #conv, roles, tokenizer, model, image_processor, context_len = get_model_llava()
    time_api = []
    time_col = []
    time_image = []
    conv, roles, tokenizer, model, image_processor, context_len = get_MGM()

    for real_file in real_files:

        if real_file["video_id"] in video_files:
            try:
                #print(real_file["video_id"])
                question = real_file["question"]
                print(question)
                prompt_ = f" Look at the image and answer this question in a concise manner in a few words: {question}"
                client = chromadb.PersistentClient(f"ego4d/data/frames/" + args.part + "/" + real_file["video_id"])
                #?collection = client.get_collection(name="mgm", embedding_function=openai_ef)
                collection = client.get_collection(name="mgm", embedding_function=openai_ef)
                time_s_collection = time.time()
                results = collection.query(
                query_texts=[question],
                n_results=2
            )
                time_e_collection = time.time()
                print("Collection processing:  ", time_e_collection-time_s_collection)
                time_col.append(time_e_collection-time_s_collection)
                frame = results['ids'][0][0]


                image = load_image(frame)
                image_size = image.size
                # print(image_size)
                image_tensor = process_images([image], image_processor, model.config)
                time_s_process = time.time()

                time_s_process = time.time()

                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                time_e_process = time.time()
                print("Image processing:  ", time_e_process-time_s_process)
                time_image.append(time_e_process-time_s_process)
                prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>" +'\n' + prompt_+ "ASSISTANT:"  
            # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                print(prompt)
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                time_s_api = time.time()
                print(image_tensor.shape)
                print(image_size)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        images_aux=None,
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=100,
                        streamer=None,
                        use_cache=False)

                outputs = tokenizer.decode(output_ids[0]).strip()
                
                time_e_api = time.time()
                print("API processing:  ", time_e_api-time_s_api)  
                time_api.append(time_e_api-time_s_api)
                
            except Exception as e:
                print(e)
            except KeyboardInterrupt:
                print(sum(time_image) / len(time_image))
                print(sum(time_api) / len(time_api))
                print(sum(time_col) / len(time_col))
                os._exit(0)


            
            print(outputs)
            
            # sample_id: {video_id,  selected frame, answer}
            temp = {"video_id": real_file["video_id"],
                    "selected_frame": frame,
                    "answer": outputs}
            temp = {"video_id": real_file["video_id"],
                    "selected_frame": frame,}
            whole[real_file["sample_id"]] = temp
            # del outputs
            # torch.cuda.empty_cache()

    json.dump(whole, open(my_file, 'w'), indent=4)
    
if __name__ =="__main__":
    main()