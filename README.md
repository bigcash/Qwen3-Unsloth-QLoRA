# Qwen3 Unsloth QLoRA

Use unsloth to finetune Qwen3 on your own dataset. **仅支持单GPU**

已成功在autodl.com的A800机器上成功微调Qwen3-32B模型。

## Dataset

> 数据请按照Qwen3的数据格式准备，示例文件：[demo_dataset.jsonl](demo_dataset.jsonl)。

**Think Data**

assistant = '<think>\n' + Reasoning + '\n</think>\n\n' + assistant

```json lines
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang?"}, {"role": "assistant", "content": "<think>\nxxx\n</think>\n\nThe capital of Zhejiang is Hangzhou."}]}
```

**No Think Data**

user = user + ' /no_think'

assistant = '\<think\>\n\n\</think\>\n\n' + assistant
```json lines
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang? /no_think"}, {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}]}
```

## Train

```shell
conda activate py310

python train.py --model_id_or_path Qwen/Qwen3-0.6B \
--data_files demo_dataset.jsonl \
--output /root/autodl-tmp/output \
--max_seq_length 512 \
--batch_size 1 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-5 \
--warmup_steps 3 \
--num_train_epochs 1 \
--rank 8 \
--lora_alpha 16 \
--seed 3407 \
--swanlab_project qwen3_unsloth \
--swanlab_mode local
```

## Jupyterlab

在Jupyter页面中，右上角切换内核中选择py310。然后开始运行代码即可。

## Inference
```shell
conda activate py310

python infer.py --model_merged /root/autodl-tmp/output/merge_4bit --max_seq_length 2048
```
