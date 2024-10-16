import socket
import chromadb
import cv2
from chromadb.utils import embedding_functions
from openai import OpenAI
import base64
import json
import numpy as np
import torch
import torch.nn.functional as F
import sys
import argparse
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from LISA.model.LISA import LISAForCausalLM
from LISA.model.llava import conversation as conversation_lib
from LISA.model.llava.mm_utils import tokenizer_image_token
from LISA.model.segment_anything.utils.transforms import ResizeLongestSide
from LISA.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import time
import os
import threading
import queue

"""
This is the code to:
1) Receive text from client
2) Retrieve an image corresponding to question
3) Generate text for answer
4) Send answer and text to the client
"""
request_queue = queue.Queue()
server_busy = False

def process_request(lisa, im_processor, transform, tokenizer, collection, all_info, folder_name):
    while True:
        conn, addr = request_queue.get()
        if conn is None:  # Poison pill means shutdown
            break
        try:
            handle_client(conn, addr, lisa, im_processor, transform, tokenizer, collection, all_info, folder_name)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
        request_queue.task_done()


def handle_client(conn, addr, lisa, im_processor, transform, tokenizer, condition, collection, all_info, folder):
    print(f"Connected by {addr}")
    global server_busy
    try:
        while True:
            if server_busy:
                print("Server is busy, ignoring new input.")
                return
            server_busy = True
            try:
                
                length_prefix = recvall(conn, 16)
                if not length_prefix:
                    return
                data_length = int(length_prefix.decode('ascii').strip())

                # Read the actual text data
                time_s = time.time()
                text_data = recvall(conn, data_length)
                if not text_data:
                    return
                text = text_data.decode('ascii')
                print(f"Received text: {text}")
                received = time.time()

                # Open the corresponding image from the server directory
                results = collection.query(
                        query_texts=[text],
                        n_results=2
                    ) 
                # print(results)
                
                img_file = results["ids"][0][0]
                #! Uncomment.
                if not os.path.isfile(img_file):
                    print(f"Error: Image file not found at {img_file}")
                    text_ = "Error: Image file not found."
                    img_file = "default.png"  # Use a default image if file is missing
                else:
                    image = cv2.imread(img_file)
                    if image is None:
                        print(f"Error: Image at {img_file} could not be loaded")
                        text_ = "Error: Image could not be loaded."
                        img_file = "default.png"  # Use a default image
                    else:
                        text_ = gpt_get_answer(img_file, text)

                        if "image" in condition:
                            img_file = get_circled(lisa, im_processor, transform, tokenizer, img_file, text_, folder)
                if "text" in condition:
                    text_ = gpt_get_answer(img_file, text)
                else:
                    text_ = text
                if "image" in condition:
                    img_file = get_circled(lisa, im_processor, transform, tokenizer, img_file, text_, folder)
                print(text)
                print(img_file)
                send_text_and_image(conn, text_, img_file)
                sent = time.time()
                all_info[text]={
                    "Received": received,
                    "sent": sent,
                    "answer": text_,
                    "img": img_file
                }
                print("Time elapsed: ", sent-time_s)
                print(text_)
                print(img_file)
            finally:
                server_busy = False 


    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        conn.close()
        json.dump(all_info, open(f"experiment/{folder}.json", 'w'), indent=4)


def start_server(lisa, im_processor, transform, tokenizer, folder_name, condition, host='0.0.0.0', port=9999):
    keys = json.load(open('my_keys.json'))

    clientGPT = OpenAI(api_key=keys["openAI"])
    MODEL="gpt-4o"

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=keys["openAI"],
                        model_name="text-embedding-ada-002"
                    )

    chroma_client = chromadb.PersistentClient("experiment/"+folder_name)
    collection = chroma_client.get_collection(name="test", embedding_function=openai_ef)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")
    all_info = {}


    try:
        while True:
            conn, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr, lisa, im_processor, transform, tokenizer, condition, collection, all_info, folder_name))
            client_thread.start()

    
    except KeyboardInterrupt:
        print("Ctrl+C pressed, shutting down the server...")

        server_socket.close()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        request_queue.put((None, None))  # Send shutdown signal
        server_socket.close()
        print("Server socket closed.")        


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8") 
    except Exception as e:
        print(f"Error encoding image: {e}")
        # Return a placeholder or default image if encoding fails
        return base64.b64encode(b"").decode("utf-8")


def gpt_get_answer(target_frame, query):
    base64_image = encode_image(target_frame)
    keys = json.load(open('my_keys.json'))
    clientGPT = OpenAI(api_key=keys["openAI"])
    response = clientGPT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a smart assistant reminding user of different things"},
            {"role": "user", "content": [
                {"type": "text", "text": f"Look at the image and answer this question briefly: {query}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        max_tokens=60,
        temperature=0.0,
    )
    return response.choices[0].message.content

def send_text_and_image(conn, text, image_path):
    # Encode text and image
    text_data = text.encode('utf-8')
    text_length = len(text_data)

    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at {image_path}")
        image_path = "default.png"
    
    # Load image and encode it as JPEG
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} could not be loaded")
        image_path = "default.png"  # Use a default image
        image = cv2.imread(image_path)


    _, image_data = cv2.imencode('.png', image)
    image_data = image_data.tobytes()
    image_length = len(image_data)

    # Create a header with lengths of text and image data
    header = f"{text_length:<16}{image_length:<16}".encode('utf-8')

    # Send header, text, and image
    conn.sendall(header)
    conn.sendall(text_data)
    conn.sendall(image_data)

def get_circled(lisa, im_processor, transform, tokenizer, img_file, text, folder):
    try:
        conv = conversation_lib.conv_templates['llava_v1'].copy()
        conv.messages = []
        prompt = "Segment according to following: " + text
        print(prompt)
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        image_path = img_file
        image_np_ = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np_, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            im_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        image_clip = image_clip.half()
        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        image = image.half()
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = lisa.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0
            # print(pred_mask)
            y_indices, x_indices = np.where(pred_mask)
            center, radius = compute_circle_params(x_indices, y_indices)

            output_image = draw_circle_on_image(image_np_, center, radius)

    # Save the result
            save_path = "{}/{}.png".format(
                "experiment/circled_"+folder, img_file.split("/")[-1].split(".")[0]
            )
            cv2.imwrite(save_path, output_image)
            return save_path
    except Exception as e:
        print(f"Error during image processing: {e}")
        return img_file



def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def compute_circle_params(x_indices, y_indices):
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))
    radius = int(np.sqrt(np.max((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)))
    return (center_x, center_y), radius


def draw_circle_on_image(image, center, radius):
    output_image = image.copy()
    cv2.circle(output_image, center, radius, (153, 255, 255), 3)  # Green color with thickness 2
    return output_image



def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def set_up_model():
    tokenizer = AutoTokenizer.from_pretrained(
        'xinlai/LISA-13B-llama2-v1',
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )          
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]  
    torch_dtype = torch.float32
    torch_dtype = torch.half
    kwargs = {"torch_dtype": torch_dtype}
    kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    model = LISAForCausalLM.from_pretrained(
        'xinlai/LISA-13B-llama2-v1', low_cpu_mem_usage=True, vision_tower="openai/clip-vit-large-patch14", seg_token_idx=seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=0)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(1024)

    model.eval()
    return model, clip_image_processor, transform, tokenizer
    
def parse_args(args):
    parser = argparse.ArgumentParser(description="server_out")
    parser.add_argument("--save", default="./temps", type=str)
    parser.add_argument("--scene", default="", type=str)
    parser.add_argument("--condition", default="", type=str)   
    return parser.parse_args(args)
                

if __name__ == "__main__":
    # client = chromadb.PersistentClient("LLM-Chroma")
    # collection = client.get_collection(name="gpt4", embedding_function=openai_ef)
    args = parse_args(sys.argv[1:])
    save_path = args.save
    scene = args.scene
    folder_name = save_path+"_"+scene
    condition = args.condition
    lisa, im_processor, transform, tokenizer = set_up_model()
    if not os.path.exists("experiment/circled_"+folder_name):
        os.makedirs("experiment/circled_"+folder_name)
    
    start_server(lisa, im_processor, transform, tokenizer, folder_name, condition)