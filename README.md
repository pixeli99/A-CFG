<div align="center">
<br>

<h3>Adaptive Classifier-Free Guidance via Dynamic Low-Confidence Masking</h3>

[Pengxiang&nbsp;Li](#)<sup>1†</sup>,
[Shilin&nbsp;Yan](#)<sup>2†♠</sup>,
[Joey&nbsp;Tsai](#)<sup>3</sup>,
[Renrui&nbsp;Zhang](#)<sup>4</sup>,
[Ruichuan&nbsp;An](#)<sup>5</sup>,  
[Ziyu&nbsp;Guo](#)<sup>4</sup>,
[Xiaowei&nbsp;Gao](#)<sup>6‡</sup>

<sup>1</sup>DUT  <sup>2</sup>Fudan  <sup>3</sup>Tsinghua  <sup>4</sup>CUHK  <sup>5</sup>PKU  <sup>6</sup>ICL

<div class="is-size-6 publication-authors">
  <p class="footnote">
    <span class="footnote-symbol"><sup>†</sup></span>Equal contribution&nbsp;&nbsp;
    <span class="footnote-symbol"><sup>♠</sup></span>Project leader&nbsp;&nbsp;
    <span class="footnote-symbol"><sup>‡</sup></span>Corresponding author
  </p>
</div>

<p align="center">
  <a href="https://github.com/pixeli99/A-CFG">
    <img src="https://img.shields.io/badge/Code-GitHub-2b3137?style=flat&logo=github&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2505.20199">
    <img src="https://img.shields.io/badge/arXiv-2505.20199-b31b1b?style=flat&logo=arXiv&logoColor=white">
  </a>
  <a href="https://arxiv.org/pdf/2505.20199">
    <img src="https://img.shields.io/badge/Paper-PDF-f6c700?style=flat&logo=adobeacrobatreader&logoColor=white">
  </a>
</p>
</div>

---
**A-CFG** is an _adaptive_ version of Classifier-Free Guidance for diffusion-based language models. Instead of a **static** unconditional input, A-CFG **dynamically re-masks low-confidence tokens at every denoising step**, focusing guidance precisely where the model is uncertain.  

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
