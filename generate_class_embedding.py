from collections import OrderedDict
from typing import Sequence
import argparse
import json
from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from transformers import AutoTokenizer, AutoConfig, XLMRobertaModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None


class XLMRobertaLanguageBackbone(nn.Module):

    def __init__(
        self,
        ckpt_path,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        init_cfg= None,
    ) -> None:

        super().__init__()
        if 'base' in ckpt_path:
            self.head = nn.Linear(768, 768, bias=True) # XLarge
            model_name = "./xlm-roberta-base/"
        elif 'large' in ckpt_path:
            self.head = nn.Linear(1024, 768, bias=True) # XLarge
            model_name = "./xlm-roberta-large/"

        self.frozen_modules = frozen_modules
        cfg = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel(cfg)
        self.language_dim = cfg.hidden_size
        
        
        # 加载 model 权重
        new_state_dict = OrderedDict()
        state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )['state_dict']
        for k, v in state_dict.items():
            if k.startswith('backbone.text_model.'):
                name = k.split("backbone.text_model.")[-1]
                new_state_dict[name] = v
        msg = self.load_state_dict(new_state_dict, strict=True)
        print(msg)

        print("TEXT-ENCODER xlm-roberta-base LOADING WEIGHTS !!!!")



    def forward(self, text: List[str]):
        text = self.tokenizer(text=text, return_tensors="pt", padding=True)
        text = text.to(device=self.model.device)
        print(text['input_ids'].shape)

        txt_feats = self.model(**text)["last_hidden_state"][:, 0]
        print(txt_feats.shape)
        txt_feats = self.head(txt_feats)
        # txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
    
        return txt_feats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wedetect_checkpoint', type=str, default='')
    parser.add_argument('--classname_file', type=str, default='data/texts/coco_zh_class_texts.json')
    args = parser.parse_args()

    with open(args.classname_file) as f:
        name_chinese = json.load(f)
    name_chinese = [name[0] for name in name_chinese]
    
    language_encoder = XLMRobertaLanguageBackbone(args.wedetect_checkpoint).cuda()
    text_embeddings = []
    num_iters = len(name_chinese) // 80 + 1 if len(name_chinese) % 80 != 0 else len(name_chinese) // 80
    with torch.no_grad():
        for i in range(num_iters):
            text_embeddings.append(language_encoder(name_chinese[i*80: (i+1)*80]))
    text_embeddings = torch.cat(text_embeddings)
    text_embeddings = F.normalize(text_embeddings, dim=-1).cpu().numpy()
    print(text_embeddings.shape)
    np.save('coco_text_embeddings.npy', text_embeddings)
