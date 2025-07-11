# Qwen3 Unsloth QLoRA

Use unsloth to finetune Qwen3 on your own dataset. **仅支持单GPU**

已成功在autodl.com的A800机器上成功微调Qwen3-32B模型。

## Build log

训练环境构建步骤：

```shell
conda create -n py310 python=3.10
conda init bash && source /root/.bashrc
conda activate py310

pip install unsloth
pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install swanlab
pip install ipykernel
python -m ipykernel install --name=py310

#sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --no-daemon
#. /root/.nix-profile/etc/profile.d/nix.sh
#nix-channel --add https://mirrors.tuna.tsinghua.edu.cn/nix-channels/nixpkgs-unstable nixpkgs
#nix-channel --update
#nix-env --file '<nixpkgs>' --install --attr llama-cpp
#
##pip install llama-cpp-python
#pip install llama_cpp_python-0.3.2-cp310-cp310-linux_x86_64.whl
```

详细的见[pip list](pip_list.md)

## Dataset

>> 数据请按照Qwen3的数据格式准备，此处仅为示例。

**Think Data**
assistant = '<think>\n' + Reasoning + '\n</think>\n\n' + assistant
```json lines
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang?"}, {"role": "assistant", "content": "<think>\nxxx\n</think>\n\nThe capital of Zhejiang is Hangzhou."}]}
```

**No Think Data**

user = user + ' /no_think'

assistant = '<think>\n\n</think>\n\n' + assistant
```json lines
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang? /no_think"}, {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}]}
```

## Train

```shell
conda activate py310

python train.py --model_id_or_path Qwen/Qwen3-0.6B \
--data_files demo1_dataset.jsonl demo2_dataset.jsonl \
--output_dir /root/autodl-tmp/output \
--max_seq_length 2048 \
--batch_size 2 \
--gradient_accumulation_steps 16 \
--learning_rate 5e-5 \
--warmup_steps 5 \
--num_train_epochs 1 \
--rank 16 \
--lora_alpha 32 \
--seed 3407 \
--swanlab_project qwen3_unsloth \
--swanlab_mode cloud
```

## Jupyterlab

在Jupyter页面中，右上角切换内核中选择py310。然后开始运行代码即可。

## Inference
```shell
conda activate py310

python infer.py --model_merged /root/autodl-tmp/output/merge_4bit --max_seq_length 2048
```
