import torch
# from memory.LLaVA.llava.utils import disable_torch_init
# from memory.LLaVA.llava.model.builder import load_pretrained_model
# from memory.LLaVA.llava.conversation import conv_templates, SeparatorStyle
from memory.MGM.mgm.utils import disable_torch_init
from memory.MGM.mgm.model.builder import load_pretrained_model
from memory.MGM.mgm.conversation import conv_templates, SeparatorStyle



def get_model_llava():
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-7b", None, "llava-v1.5-7b", False, True, device="cuda")
    
    conv = conv_templates["llava_v1"].copy()
    roles = conv.roles
    return conv, roles, tokenizer, model, image_processor, context_len


def get_MGM():
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model("memory/MGM/MGM-7B", None, "MGM-7B", False, True, device="cuda")
    conv = conv_templates["vicuna_v1"].copy()
    roles = conv.roles
    return conv, roles, tokenizer, model, image_processor, context_len