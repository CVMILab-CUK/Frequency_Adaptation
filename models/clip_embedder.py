import torch 
import torch.nn as nn
import open_clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch.nn.functional as F


class Frozen_CLIPImageNTextEmbedder(nn.Module):
    def __init__(self, 
                 model_id='ViT-H-14', 
                 pretrained='laion2b_s32b_b79k', 
                 prompt =  "a high-quality photo of a face",
                 force_custom_text=True,
                ):
        super().__init__()
        self.model, self.train_transform, self.eval_transform = open_clip.create_model_and_transforms(model_id, pretrained=pretrained, force_custom_text=force_custom_text)
        self.model.eval()
        self.text_encoder = self.model.text
        self.tokenizer = open_clip.get_tokenizer(model_id)
        for p in self.model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            tokens = self.tokenizer(prompt)
            text_embeds = self.model.text.token_embedding(tokens)
            text_embeds = text_embeds + self.model.text.positional_embedding
            text_embeds = self.model.text.transformer(text_embeds)
            text_embeds = self.model.text.ln_final(text_embeds)
            self.register_buffer('static_text_embeds', text_embeds)

    @torch.no_grad() 
    def __call__(self, img: torch.tensor, ori_img=None):    
        clip_image_inputs = self.eval_transform(img)        
        image_embeds = self.model.encode_image(clip_image_inputs) # [B, D]

        batch_size = img.shape[0]
        text_embeds = self.static_text_embeds.repeat(batch_size, 1, 1)
        return text_embeds, image_embeds