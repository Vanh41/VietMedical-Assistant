import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import Dataset
from unsloth.chat_templates import train_on_responses_only


class FineTuner:
    def __init__(
        self,
        model_name: str,
        max_seq_length = 2048,
        dtype = None, 
        load_in_4bit: Bool = True,
        ):
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
    def add_lora(
        self,
        r = 32, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        ):
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = r, 
            target_modules = target_modules,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout, 
            bias = bias,    
            use_gradient_checkpointing = use_gradient_checkpointing,
            random_state = random_state,
            use_rslora = use_rslora,  
            loftq_config = loftq_config, 
        )
        
        return self.model


    def train(self, train_dataset):
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = train_dataset,
            dataset_text_field = "text",
            max_seq_length = 4096,
            packing = False, # Can make training 5x faster for short sequences.
            args = SFTConfig(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                num_train_epochs = 1, # Set this for 1 full training run.
                learning_rate = 2e-4,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none", # Use TrackIO/WandB etc
            ),
        )
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
            )

        trainer_stats = trainer.train()
        return trainer_stats
          
    def generate(self, prompt: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save_model(self, path: str, save_method: str = "merged_16bit"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
