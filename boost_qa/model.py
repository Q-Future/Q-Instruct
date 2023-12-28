### Install mPLUG-Owl from https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2#install

import torch.nn as nn
import torch

from typing import List
from PIL import Image

from mplug_owl2.model.builder import load_pretrained_model

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class QInstructScorer(nn.Module):
    def __init__(self, boost=True, device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model("q-future/q-instruct-mplug-owl2", None, "mplug_owl2", device=device)
        prompt = "USER: <|image|>Rate the quality of the image.\nASSISTANT: "
        
        if not boost:
            self.boost = False
            self.preferential_ids_ = [id_[1] for id_ in tokenizer(["good", "average", "poor"])["input_ids"]]
            self.weight_tensor = torch.Tensor([1, 0.5, 0]).half().to(model.device)
        else:
            self.boost = True
            self.preferential_ids_ = [id_[1] for id_ in tokenizer(["good", "average", "poor", "high", "medium", "low", "fine", "acceptable", "bad"])["input_ids"]]
            self.weight_tensor = torch.Tensor([1, 0.5, 0]).half().to(model.device)
    
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
    def forward(self, image: List[Image.Image]):
        with torch.inference_mode():
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            output_logits = self.model(self.input_ids.repeat(image_tensor.shape[0], 1),
                        images=image_tensor)["logits"][:,-1, self.preferential_ids_]
            if self.boost:
                output_logits = output_logits.reshape(-1, 3, 3).mean(1)
            return torch.softmax(output_logits, -1) @ self.weight_tensor
    

if __name__ == "__main__":
    scorer = QInstructScorer(boost=False)
    print(scorer([Image.open("fig/examples_211.jpg"),Image.open("fig/sausage.jpg")]).tolist())
