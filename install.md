## Build log

训练环境构建步骤：

```shell
apt update
apt upgrade
apt install tmux

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

pip list:

```text
Package                  Version
------------------------ -----------
accelerate               1.8.1
aiohappyeyeballs         2.6.1
aiohttp                  3.12.13
aiosignal                1.4.0
annotated-types          0.7.0
asttokens                3.0.0
async-timeout            5.0.1
attrs                    25.3.0
bitsandbytes             0.46.1
boto3                    1.39.3
botocore                 1.39.3
certifi                  2025.6.15
charset-normalizer       3.4.2
click                    8.2.1
comm                     0.2.2
cut-cross-entropy        25.1.1
datasets                 3.6.0
debugpy                  1.8.14
decorator                5.2.1
diffusers                0.34.0
dill                     0.3.8
docstring_parser         0.16
einops                   0.8.1
exceptiongroup           1.3.0
executing                2.2.0
filelock                 3.18.0
flash_attn               2.8.0.post2
frozenlist               1.7.0
fsspec                   2025.3.0
hf_transfer              0.1.9
hf-xet                   1.1.5
huggingface-hub          0.33.2
idna                     3.10
importlib_metadata       8.7.0
ipykernel                6.29.5
ipython                  8.37.0
jedi                     0.19.2
Jinja2                   3.1.6
jmespath                 1.0.1
jupyter_client           8.6.3
jupyter_core             5.8.1
markdown-it-py           3.0.0
MarkupSafe               3.0.2
matplotlib-inline        0.1.7
mdurl                    0.1.2
mpmath                   1.3.0
msgspec                  0.19.0
multidict                6.6.3
multiprocess             0.70.16
nest-asyncio             1.6.0
networkx                 3.4.2
numpy                    2.2.6
nvidia-cublas-cu12       12.6.4.1
nvidia-cuda-cupti-cu12   12.6.80
nvidia-cuda-nvrtc-cu12   12.6.77
nvidia-cuda-runtime-cu12 12.6.77
nvidia-cudnn-cu12        9.5.1.17
nvidia-cufft-cu12        11.3.0.4
nvidia-cufile-cu12       1.11.1.6
nvidia-curand-cu12       10.3.7.77
nvidia-cusolver-cu12     11.7.1.2
nvidia-cusparse-cu12     12.5.4.2
nvidia-cusparselt-cu12   0.6.3
nvidia-ml-py             12.575.51
nvidia-nccl-cu12         2.26.2
nvidia-nvjitlink-cu12    12.6.85
nvidia-nvtx-cu12         12.6.77
packaging                25.0
pandas                   2.3.1
parso                    0.8.4
peft                     0.16.0
pexpect                  4.9.0
pillow                   11.3.0
pip                      25.1
platformdirs             4.3.8
prettytable              3.16.0
prompt_toolkit           3.0.51
propcache                0.3.2
protobuf                 3.20.3
psutil                   7.0.0
ptyprocess               0.7.0
pure_eval                0.2.3
pyarrow                  20.0.0
pydantic                 2.11.7
pydantic_core            2.33.2
pyecharts                2.0.8
Pygments                 2.19.2
pynvml                   12.0.0
python-dateutil          2.9.0.post0
pytz                     2025.2
PyYAML                   6.0.2
pyzmq                    27.0.0
regex                    2024.11.6
requests                 2.32.4
rich                     13.9.4
s3transfer               0.13.0
safetensors              0.5.3
sentencepiece            0.2.0
setuptools               78.1.1
shtab                    1.7.2
simplejson               3.20.1
six                      1.17.0
stack-data               0.6.3
swankit                  0.2.4
swanlab                  0.6.6
sympy                    1.14.0
tokenizers               0.21.2
torch                    2.7.0
torchvision              0.22.0
tornado                  6.5.1
tqdm                     4.67.1
traitlets                5.14.3
transformers             4.53.1
triton                   3.3.0
trl                      0.19.1
typeguard                4.4.4
typing_extensions        4.14.1
typing-inspection        0.4.1
tyro                     0.9.26
tzdata                   2025.2
unsloth                  2025.6.12
unsloth_zoo              2025.6.8
urllib3                  2.5.0
wcwidth                  0.2.13
wheel                    0.45.1
wrapt                    1.17.2
xformers                 0.0.30
xxhash                   3.5.0
yarl                     1.20.1
zipp                     3.23.0
```
