from transformers import AutoModel, AutoTokenizer
import torch

import random
from copy import deepcopy
import json
from tqdm import tqdm
from decord import VideoReader
from PIL import Image

torch.set_grad_enabled(False)


import os
os.makedirs("results/mix-internlm_xcomposer_vl/",exist_ok=True)
torch.manual_seed(1234)

# init model and tokenizer
model = AutoModel.from_pretrained('DLight1551/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('DLight1551/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True)
model.tokenizer = tokenizer

print(tokenizer(["good"]), tokenizer.decode(5527))

image_paths = [
        "../datasets/KoNViD_1k_videos/",
    ]

json_prefix = "../datasets/json/"
jsons = [
        json_prefix + "konvid.json",
    ]

def load_video(video_file):
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]

    return [Image.fromarray(vr[index].asnumpy()) for index in frame_indices]

def get_logits(model, text, image):
    with torch.cuda.amp.autocast():
        image = model.vis_processor(image).unsqueeze(0).to(model.device)
        img_embeds = model.encode_img(image)
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [ 
        model.tokenizer(seg,
                             return_tensors='pt',
                             add_special_tokens=i == 0). 
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]   
    prompt_seg_embs = [ 
        model.internlm_model.model.embed_tokens(seg)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    
    return model.internlm_model(
        inputs_embeds=prompt_embs).logits[:,-1]


for image_path, input_json in zip(image_paths, jsons):
    with open(input_json) as f:
        iqa_data = json.load(f)
    
    for i, llddata in enumerate(tqdm(iqa_data, desc=image_path)):
        message = "Rate the quality of the image."
        
        images = load_video(image_path + llddata["img_path"])
        llddata["logit_good"] = 0
        llddata["logit_poor"] = 0

        for image in images:
            # 1st dialogue turn
            output_logits = get_logits(model, message, image)
            probs, inds = output_logits.sort(dim=-1,descending=True)
            lgood, lpoor = output_logits[0,18682].item(), output_logits[0,5527].item()
            llddata["logit_good"] += lgood
            llddata["logit_poor"] += lpoor

        with open(f"results/mix-internlm_xcomposer_vl/{input_json.split('/')[-1]}", "a") as wf:
            json.dump(llddata, wf)