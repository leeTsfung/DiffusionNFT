

<h1 align="center"> DiffusionNFT: 基于前向过程的在线扩散强化学习 </h1>
<div align="center">
<a href='https://arxiv.org/abs/2509.16117'><img src='https://img.shields.io/badge/论文_(arXiv)-2509.16117-red?logo=arxiv'></a> &nbsp;
<a href='https://research.nvidia.com/labs/dir/DiffusionNFT'><img src='https://img.shields.io/badge/官网-green?logo=homepage&logoColor=white'></a> &nbsp;
<a href='https://huggingface.co/worstcoder/SD3.5M-DiffusionNFT-MultiReward'><img src='https://img.shields.io/badge/模型-blue?logo=huggingface&logoColor='></a> &nbsp;
</div>

算法概览

DiffusionNFT 是一种针对扩散模型的全新在线强化学习范式，它直接在前向扩散过程中执行策略优化。

求解器无关 (Solver-Agnostic)： 与 GRPO 不同，DiffusionNFT 在整个数据收集过程中兼容任何黑盒采样器（例如高阶 ODE 求解器）。

理论一致且显存高效 (Theoretically Consistent & Memory Efficient)： 通过在前向过程中操作，DiffusionNFT 保持了前向一致性，并且在训练时仅需要干净的图片，而不需要存储整个采样轨迹。

简单且兼容性强 (Simple & Compatible)： DiffusionNFT 建立在标准的流匹配 (flow-matching) 目标之上，使其易于集成到现有的扩散模型训练代码库中。

<p align="center">
<img src="./assets/performance.png" alt="结果展示" style="width:90%;">
</p>


DiffusionNFT 的流程包括：

数据收集： 当前的采样策略 
𝑣
old
v
old
 生成图片，这些图片由奖励函数进行评估。

概念性数据拆分： 图片根据其奖励分数被概念性地拆分为正样本集和负样本集。

前向过程优化： 训练策略 
𝑣
𝜃
v
θ
	​

 在收集到的图片的加噪版本上进行优化。我们新颖的损失函数利用奖励分数在隐式的正向和负向目标之间进行加权，从而直接将强化信号整合到模型参数中。

<p align="center">
<img src="./assets/method.png" alt="DiffusionNFT 方法" style="width:80%;">
</p>

环境配置

我们的实现基于 Flow-GRPO 代码库，大部分环境保持一致。

通过以下命令克隆本仓库并安装依赖包：

code
Bash
download
content_copy
expand_less
git clone https://github.com/NVlabs/DiffusionNFT.git
cd DiffusionNFT

conda create -n DiffusionNFT python=3.10.16
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
奖励模型准备

我们支持的奖励模型包括 GenEval、OCR、PickScore、ClipScore、HPSv2.1、Aesthetic、ImageReward 和 UnifiedReward。此外，我们在 FlowGRPO 的基础上增加了对 HPSv2.1 的支持，并将 GenEval 从远程服务器简化为本地运行。

下载 Checkpoints (权重文件)
code
Bash
download
content_copy
expand_less
mkdir reward_ckpts
cd reward_ckpts
# Aesthetic (美学评分)
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
# GenEval (综合评估)
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
# ClipScore (图文匹配)
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
# HPSv2.1 (人类偏好)
wget https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
cd ..
安装奖励环境依赖
code
Bash
download
content_copy
expand_less
# GenEval 环境
pip install -U openmim
mim install mmengine
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout 1.x
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -e . -v
cd ..

pip install open-clip-torch clip-benchmark

# OCR 环境
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein

# HPSv2.1 环境
pip install hpsv2x==1.2.0

# ImageReward 环境
pip install image-reward
pip install git+https://github.com/openai/CLIP.git

对于 UnifiedReward，我们使用 sglang 部署奖励服务。为了避免冲突，请先创建一个新环境并安装 sglang：

code
Bash
download
content_copy
expand_less
pip install "sglang[all]"

然后启动服务：

code
Bash
download
content_copy
expand_less
python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85

可以通过降低 --mem-fraction-static、限制 --max-running-requests 以及增加 --data-parallel-size 或 --tensor-parallel-size 来减少显存使用。

训练

与 FlowGRPO 不同，我们使用 torchrun 而不是 accelerate 来进行分布式训练。默认的配置文件 config/nft.py 是为 8 张 GPU 设置的，您可以根据需要进行自定义。

单节点训练示例：

code
Bash
download
content_copy
expand_less
export WANDB_API_KEY=xxx
export WANDB_ENTITY=xxx

# GenEval 任务
torchrun --nproc_per_node=8 scripts/train_nft_sd3.py --config config/nft.py:sd3_geneval

# 多奖励 (Multi-reward) 任务
torchrun --nproc_per_node=8 scripts/train_nft_sd3.py --config config/nft.py:sd3_multi_reward
评估

我们提供了一个推理脚本，用于加载 LoRA checkpoints 并运行评估。

code
Bash
download
content_copy
expand_less
# Hugging Face LoRA checkpoint, 开启 CFG
torchrun --nproc_per_node=8 scripts/evaluation.py \
    --lora_hf_path "jieliu/SD3.5M-FlowGRPO-GenEval" \
    --model_type sd3 \
    --dataset geneval \
    --guidance_scale 4.5 \
    --mixed_precision fp16 \
    --save_images

# 本地 LoRA checkpoint, 无需 CFG (w/o CFG)
torchrun --nproc_per_node=8 scripts/evaluation.py \
    --checkpoint_path "logs/nft/sd3/geneval/checkpoints/checkpoint-1018" \
    --model_type sd3 \
    --dataset geneval \
    --guidance_scale 1.0 \
    --mixed_precision fp16 \
    --save_images

--dataset 标志支持 geneval、ocr、pickscore 和 drawbench。

致谢

感谢 Flow-GRPO 项目提供了很棒的开源扩散强化学习代码库。

引用
code
Code
download
content_copy
expand_less
@article{zheng2025diffusionnft,
  title={DiffusionNFT: Online Diffusion Reinforcement with Forward Process},
  author={Zheng, Kaiwen and Chen, Huayu and Ye, Haotian and Wang, Haoxiang and Zhang, Qinsheng and Jiang, Kai and Su, Hang and Ermon, Stefano and Zhu, Jun and Liu, Ming-Yu},
  journal={arXiv preprint arXiv:2509.16117},
  year={2025}
}
