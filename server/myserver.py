import socket
import struct
import numpy as np
import cv2
import threading
import queue
import time
from openai import OpenAI
# from llava.utils import disable_torch_init
# import transformers

# import chromadb
# from transformers import AutoProcessor, AutoModelForCausalLM
# from transformers import BitsAndBytesConfig
# from transformers import pipeline
from PIL import Image
import glob
import torch
import json
from utils import check_blur, similar, apply_wiener_filter, interpolate_frames
import chromadb
from chromadb.utils import embedding_functions
import base64
import argparse
import sys
import os

"""
This code:
1) Receives captured images from the client, filters and saves them
2) Generates textual description for each saved image
3) Saves description together with the link to the file in a Vector DB
"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_data(image, folder, collection): #! process_data(image, pipe):
    """
    1)Check if frame is blurry; if is, discard;
    2) Check if frame is similar to the previous; if is, discard!    
    """
    nparr = np.frombuffer(image, np.uint8)

    # Decode the image data to an OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to decode image")
        return
    # img = apply_wiener_filter(img)
    is_blurry = check_blur(img)
    if is_blurry: 
        # print("Blurry!!!")
        #filename = f"temps/all/{int(time.time())}_blurry_frame.png"
        return
    filename = f"experiment/{folder}/{int(time.time())}_frame.png"

    if not frames:
        cv2.imwrite(filename, img)
        frames.append(filename)
        print("saved as first!!!")
        return
    img_to_compare = cv2.imread(frames[-1])
    is_similar = similar(img_to_compare, img)
    if is_similar:
        print("Similar!!!!")
        
        return
    cv2.imwrite(filename, img)
    frames.append(filename)
    print("saved, different from last!!!")
    #? Language processing! 
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image=Image.fromarray(color_coverted)
    # outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 400})
    base64_image = encode_image(filename)
    response = clientGPT.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": role},
        {"role": "user", "content": [
            {"type": "text", "text": prompt
            },
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    max_tokens=400,
    temperature=0.0,
)

    collection.add(
        ids=[filename],
        documents=[response.choices[0].message.content] # A list of numpy arrays representing images
    )
    # del outputs
    # torch.cuda.empty_cache()




def start_server(folder, collection, host='0.0.0.0', port=9999): #! 117.17.187.152 start_server(pipe, host='0.0.0.0', port=9999):
    my_data = dict()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    try:
        while True:
            # Read the length prefix
            length_prefix = recvall(conn, 16)
            if not length_prefix:
                break
            data_length = int(length_prefix.decode('ascii').strip())

            # Read the actual image data
            image_data = recvall(conn, data_length)
            if not image_data:
                break

            # Convert the image data to a numpy array
            process_data(image_data, folder, collection) #!process_data(image_data, pipe)
            

    except KeyboardInterrupt:
        print("Ctrl+C pressed, closing the server...")
        conn.close()
        server_socket.close()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
        server_socket.close()

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def parse_args(args):
    parser = argparse.ArgumentParser(description="server_in")
    parser.add_argument("--save", default="./temps", type=str)
    parser.add_argument("--scene", default="", type=str)

    return parser.parse_args(args)

if __name__ == "__main__":
    frames = []
    args = parse_args(sys.argv[1:])
    save_path = args.save
    scene = args.scene
    folder_name = save_path+"_"+scene
    if not os.path.exists("experiment/"+folder_name):
        os.makedirs("experiment/"+folder_name)
    keys = json.load(open('my_keys.json'))

    clientGPT = OpenAI(api_key=keys["openAI"])
    MODEL="gpt-4o"

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=keys["openAI"],
                        model_name="text-embedding-ada-002"
                    )

    chroma_client = chromadb.PersistentClient("experiment/"+folder_name)
    collection = chroma_client.create_collection(
        name="test",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    prompt = "Describe the image in detail. Start with a high-level description: Begin by providing an overall description of the image, capturing its main subject or scene. Describe the visual elements: Break down the image into its key visual elements and describe them in detail, including color: start from left part of the picture and proceed to the right, mentioning what is placed next to what or under/on what. Finally, avoid fabricating information."
    role = "You are a smart assistant generating interior descriptions"
    # prompt = f"USER: <image>\n{inp}\nASSISTANT:"

    my_data=dict()
    
    start_server(folder_name, collection)
    #? Activate for textual processing!
    