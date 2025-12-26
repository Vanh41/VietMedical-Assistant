# VietMedical-Assistant

*Fine-tuning Qwen 3-4B-Instruct model for medical Question-Answer dataset*

This is a personal project I developed with minimal help from LLM for the UI. I created it out of my own interest (っ˘ω˘ς )❤️.


---

## Table of Contents

* 1\. [ Overview 🚀](#Overview)
* 2\. [ Usage 🔑](#Usage)
  * 2.1. [Installation](#Dependency)
  * 2.2. [To run the project](#Toruntrainingpipeline)


##  1. <a name='Overview'></a> Overview 🚀

- This repository contains the work for fine-tuning a model with custom dataset. My approach is to leverage Qwen 3-4B-Instruct via a two stage training pipeline: *Supervised Fine-Tuning (SFT)* followed
by *Direct Preference Optimization (DPO)* to ensure terminological accuracy and naturalness in the medical domain.
- For the dataset, I use https://huggingface.co/datasets/lqkhoi/viet_med_qa for the SFT phase. For the synthetic data in DPO phase, I randomly extract 1k sample from the SFT data, then use Mistral-medium-lastest
to refine the answer.

##  2. <a name='Usage'></a> Usage 🔑

- You can find the result of some traditional statistical metrics in `./Eval_Comparison.json`
- You can also check the training results in [WandB](https://wandb.ai/vietanhm6a-hanoi-university-of-science-and-technology/Medical-Chatbot?nw=nwuservietanhm6a)

####  2.1. <a name='Dependency'></a>Installation
After cloning the repo, you can install dependencies locally on Python>=3.11 as follows:

```bash
pip install -r requirements.txt
```

####  2.2. <a name='Toruntrainingpipeline'></a>To run the project
If you want to check my training process, you can check out `./notebooks`.
In case you want to try my results:
```bash
python app.py
```
And then visit `http://127.0.0.1:8000` to test


