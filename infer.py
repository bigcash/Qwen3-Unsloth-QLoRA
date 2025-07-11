import argparse
from transformers import TextStreamer
from unsloth import FastLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_merged", type=str, default="")
parser.add_argument("--max_seq_length", type=int, default=4096)
args = parser.parse_args()


if __name__ == '__main__':
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_merged,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    prompt = "Solve (x + 2)^2 = 0."
    print("prompt: "+prompt)
    print("==================No Thinking==============")
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
        enable_thinking=False,  # Disable thinking
    )
    result = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=256,  # Increase for longer outputs!
        temperature=0.7, top_p=0.8, top_k=20,  # For non thinking
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    print(result)

    print("==================Thinking==============")

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
        enable_thinking=True,  # Enable thinking
    )
    result = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=1024,  # Increase for longer outputs!
        temperature=0.6, top_p=0.95, top_k=20,  # For thinking
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    print(result)
