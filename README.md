# VietMedical-Assistant

*Fine-tuning Qwen 3-4B-Instruct model for medical Question-Answer dataset*

This is a personal project I developed with minimal help from LLM for the UI. I created it out of my own interest.

---

## Table of Contents

* 1\. [ Overview 🚀](#Overview)
* 2\. [ Usage 🔑](#Usage)
  * 2.1. [Installation](#Dependency)
  * 2.2. [To run the project](#Toruntrainingpipeline)

## 1. `<a name='Overview'></a>` Overview 🚀

- This repository contains the work for fine-tuning a model with custom dataset. My approach is to leverage Qwen 3-4B-Instruct via a two stage training pipeline: *Supervised Fine-Tuning (SFT)* followed
  by *Direct Preference Optimization (DPO)* to ensure terminological accuracy and naturalness in the medical domain.
- For the dataset, I use https://huggingface.co/datasets/hungnm/vietnamese-medical-qa for the SFT phase. For the synthetic data in DPO phase, I randomly extract 3k sample from the SFT data, with the answers of dataset to be the ground truth and use the response with high temperature of SFT model to be the rejected one.

## 2. `<a name='Usage'></a>` Usage 🔑

- You can find the result of some traditional statistical metrics in `/evaluation`
- You can also check the training results in [WandB](https://wandb.ai/vietanhm6a-hanoi-university-of-science-and-technology/Medical-Chatbot?nw=nwuservietanhm6a)

#### 2.1. `<a name='Dependency'></a>`Installation

After cloning the repo, you can install dependencies locally on Python>=3.11 as follows:

```bash
pip install -r requirements.txt
```

#### 2.2. `<a name='Toruntrainingpipeline'></a>`To run the project

If you want to check my training process, you can check out `./notebooks`.
In case you want to try my results:

```bash
python app.py
```

And then visit `http://127.0.0.1:8000` to test

<img width="1512" height="982" alt="Screenshot 2025-12-15 at 10 25 58" src="https://github.com/user-attachments/assets/119b4678-694d-4c66-b4bc-bc37262225e0" />
<img width="1512" height="982" alt="Screenshot 2025-12-15 at 10 25 14" src="https://github.com/user-attachments/assets/baf86b33-a8cd-47be-b1ed-4ba398ef124c" />

https://github.com/user-attachments/assets/0c94a1e5-19b3-4b16-a35b-a092633cfd69
