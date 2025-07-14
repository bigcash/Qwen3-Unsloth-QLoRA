import argparse
import os
import time

import pandas as pd
import torch
from datasets import Dataset
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_id_or_path", type=str, default="")
parser.add_argument("--output", type=str, default="output")
parser.add_argument('--data_files', nargs='+', type=str, default=[], help='qwen3 data files, e.g., --data_files demo1.jsonl demo2.jsonl')
parser.add_argument('--target_modules', nargs='+', type=str, default=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"], help='lora target modules')
parser.add_argument("--max_seq_length", type=int, default=16384)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument("--logging_steps", type=int, default=1)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--rank", type=int, default=4)
parser.add_argument("--lora_alpha", type=int, default=8)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--swanlab_project", type=str, default="qwen3_unsloth")
parser.add_argument("--swanlab_experiment_name", type=str, default="")
parser.add_argument("--swanlab_mode", type=str, default="cloud", help="local|cloud|offline|disabled")
args = parser.parse_args()


def main():
    os.makedirs(args.output, exist_ok=True)
    lora_model = os.path.join(args.output, "lora_model")
    merge_4bit = os.path.join(args.output, "merge_4bit")
    merge_4bit_gguf = os.path.join(args.output, "gguf-q4_k_m")
    # os.environ["SWANLAB_PROJECT"] = args.swanlab_project
    max_seq_length = args.max_seq_length
    assert len(args.data_files) > 0, "data_files must not be empty"
    if not args.swanlab_experiment_name:
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        args.swanlab_experiment_name = time_str

    swanlab_callback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment_name,
        mode=args.swanlab_mode,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id_or_path,
        max_seq_length=max_seq_length,  # Context length - can be longer, but uses more memory
        load_in_4bit=True,  # 4bit uses much less memory
        load_in_8bit=False,  # A bit more accurate, uses 2x memory
        full_finetuning=False,  # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,  # Best to choose alpha = rank or rank*2
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=args.seed,
        max_seq_length=max_seq_length,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    conversations = []
    for df in args.data_files:
        dataset0 = load_dataset("json", data_files=df, split='train')
        conversations0 = tokenizer.apply_chat_template(
            dataset0["messages"],
            tokenize=False,
        )
        conversations.append(conversations0)
    data = pd.concat([
        pd.Series(c) for c in conversations
    ])
    data.name = "text"

    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed=args.seed)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        train_dataset=combined_dataset,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # Use GA to mimic batch size!
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,  # Set this for 1 full training run.
            # max_steps=30,
            learning_rate=args.learning_rate,  # Reduce to 2e-5 for long training runs
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="swanlab",  # Use this for WandB etc
        ),
        callbacks=[swanlab_callback],
    )
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained(lora_model)  # Local saving
    tokenizer.save_pretrained(lora_model)

    # load lora model
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=lora_model,  # YOUR MODEL YOU USED FOR TRAINING
    #     max_seq_length=max_seq_length,
    #     load_in_4bit=True,
    # )

    # Merge to 4bit
    model.save_pretrained_merged(merge_4bit, tokenizer, save_method="merged_4bit_forced", )
    # Save to q4_k_m GGUF, need install llama.cpp
    # model.save_pretrained_gguf(merge_4bit_gguf, tokenizer, quantization_method="q4_k_m")


if __name__ == '__main__':
    main()
