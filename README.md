---
license: mit
language:
- en
metrics:
- f1
- accuracy
pipeline_tag: visual-question-answering
tags:
- agriculture
- tomato
- leaf
- disease
- vqa
- mixture-of-experts
- multimodal
library_name: transformers
---

# 🍅 SOLAR: A Multi-Modal Generative Model for Tomato Disease Leaves Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2609.19555-b31b1b.svg)](https://arxiv.org/abs/2609.19555)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-enalis%2FSOLAR-blue)](https://huggingface.co/enalis/SOLAR)
[![Hugging Face SCOLD](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-enalis%2Fscold-green)](https://huggingface.co/enalis/scold)

**SOLAR** is a multimodal generative model that formulates tomato leaf disease diagnosis as a **Visual Question Answering (VQA)** task. By integrating the domain-specific **SCOLD vision foundation model** with a **LiquidAI LFM2.5 language backbone** via a task-aware **Image Mixture-of-Experts (ImageMoE)** and **Residual Fusion**, SOLAR enables comprehensive multimodal reasoning across diverse diagnostic tasks, including symptom recognition, disease identification, severity assessment, and treatment recommendations.

---

### ✅ Intended Use
- Generative Visual Question Answering (VQA) for tomato leaf disease diagnosis
- Multi-task plant pathology reasoning (symptom recognition, disease identification, severity assessment)
- Interactive multimodal AI assistant for smart farming and precision agriculture

---

## 🧪 How to Use

First clone our repository:

```bash
git clone https://github.com/EnalisUs/SOLAR.git
cd SOLAR
pip install -r requirements.txt
```

Clone the **SCOLD** vision foundation model:

```bash
git clone https://huggingface.co/enalis/scold
```

Download our pretrained model weights and LoRA adapter into `checkpoints/`:

```bash
mkdir -p checkpoints
python -c "
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download(repo_id='enalis/SOLAR', filename='model_epoch_19.pth', local_dir='checkpoints')
snapshot_download(repo_id='enalis/SOLAR', allow_patterns='lora_epoch_19/*', local_dir='checkpoints')
"
```

Please find detail to load and evaluate our model in *test.py*:

```bash
python test.py
```

Or run inference in Python:

```python
import os
import torch
from PIL import Image
from torchvision import transforms
from model import SCOLD_LFM_ImageMoE_LoRA

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize model
model = SCOLD_LFM_ImageMoE_LoRA(
    scold_ckpt="scold/scold.pth",
    num_experts=4,
    top_k=2,
    use_lora=True,
).to(device)

# 2. Load trained checkpoint
ckpt = torch.load("checkpoints/model_epoch_19.pth", map_location=device)
model.visual_proj.load_state_dict(ckpt["visual_proj"])
model.visual_text_fusion.load_state_dict(ckpt["fusion"])
model.image_moe.load_state_dict(ckpt["image_moe"])

# Load LoRA adapter if available
if os.path.exists("checkpoints/lora_epoch_19"):
    model.text_model.load_adapter("checkpoints/lora_epoch_19", adapter_name="default")
    model.text_model.set_adapter("default")

model.eval()

# 3. Preprocess image & question
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Single image tensor [3, 224, 224]
image = transform(Image.open("path_to_leaf.jpg").convert("RGB")).to(device)
question = "What disease is affecting this tomato leaf?"

# 4. Generate diagnosis
with torch.no_grad():
    answer = model.generate(image, question, max_tokens=20)
    print(f"Answer: {answer}")
```

---

Please cite this paper if this code is useful for you!

```bibtex
@article{quoc2026solar,
  title={A Multi-Modal Generative Model for Tomato Disease Leaves Understanding},
  author={Quoc, Khang Nguyen and Tran, Minh-Phuoc and Truong, Gia-Han and Quach, Luyl-Da},
  journal={arXiv preprint arXiv:2609.19555},
  year={2026},
  url={https://arxiv.org/abs/2609.19555}
}
```

