# Adaptive Classifier-Free Guidance (A-CFG)

**A-CFG** is an _adaptive_ version of Classifier-Free Guidance for diffusion-based language models.  
Instead of a **static** unconditional input, A-CFG **dynamically re-masks low-confidence tokens at every denoising step**, focusing guidance precisely where the model is uncertain.  

> **Adaptive Classifier-Free Guidance via Dynamic Low-Confidence Masking**  
> Pengxiang Li\*, Shilin Yan\*, Joey Tsai, Ray Zhang, Ruichuan An, Ziyu Guo, Xiaowei Gao  
> [[arXiv](https://arxiv.org/abs/xxxx.xxxxx)]

---

## ✨ Key Features
* **Plug-and-play** guidance module for any masked diffusion language model (e.g. LLaDA, Dream).
* **Token-level confidence heuristics** with a single hyper-parameter `ρ` (remask ratio).

---

## 🚀 Quick Start

This project builds on LLaDA. See their [README](https://github.com/ML-GSAI/LLaDA/blob/main/README.md) for more details on the base model setup.

### LLaDA Inference
The [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) are uploaded
in Huggingface. Please first install `transformers==4.38.2` and employ the [transformers](https://huggingface.co/docs/transformers/index) to load.

```angular2html
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```
