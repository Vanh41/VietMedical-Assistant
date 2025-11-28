# train.py
import argparse
from model.unsloth_model import FineTuner
from src.data.data_loader import prepare_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-1B-bnb-4bit")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    # Load data
    dataset = prepare_dataset(args.data_path)
    # Init model
    ft = FineTuner(args.model_name)
    # Add LoRA
    ft.add_lora(r=32)
    # Train
    trainer_stats = ft.train(
        train_dataset=dataset,
    )
    # Save
    ft.save_model(f"{args.output_dir}/final")
    print("Training completed!")

if __name__ == "__main__":
    main()