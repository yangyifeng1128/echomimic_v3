# EchoMimic V3: 13亿参数即可实现统一多模态、多任务人体动画生成

# 一、快速开始

## 1.1 部署环境
- 操作系统：Ubuntu 22.04, CUDA 12.4
- GPU：H800 (80GB)
- Python：3.11

## 1.2 安装

### 1.2.1 安装依赖包

```sh
uv sync
```

### 1.2.2 准备模型

| 模型名称            | 下载链接                                                                     | 备注              |
| ------------------- | ---------------------------------------------------------------------------- | ----------------- |
| Wan2.1-Fun-1.3B-InP | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | 基础模型          |
| wav2vec2-base       | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | 音频编码器        |
| EchoMimicV3-preview | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                | EchoMimic V3 权重 |

在终端下，执行以下命令，下载模型：

```sh
mkdir models
cd models

uv run hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP --local-dir ./Wan2.1-Fun-V1.1-1.3B-InP
uv run hf download facebook/wav2vec2-base-960h --local-dir ./wav2vec2-base-960h
uv run hf download BadToBest/EchoMimicV3 --local-dir ./
```

以下是下载完成的模型文件目录结构：

```sh
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

## 1.3 开始推理

```
uv run app_mm.py
```

### 1.3.1 提示信息

> - 音频 CFG：音频 CFG `audio_guidance_scale` 最佳范围为 2~3。增加音频 CFG 值可以改善唇同步效果，减少音频 CFG 值可以提高视觉质量。
> - 文本 CFG：文本 CFG `guidance_scale` 最佳范围为 3~6。增加文本 CFG 值可以更好地遵循提示词，减少文本 CFG 值可以提高视觉质量。
> - TeaCache：`teacache_threshold` 的最佳范围为 0~0.1。
> - 采样步数：头部动画为 5 步，全身动作为 15~25 步。
> - ​长视频生成：如果需要生成超过 138 帧的视频，可以使用长视频 CFG。
> - 尝试降低 `partial_video_length` 节省显存。

## 1.4 暴露多个端口

参考资料：
- https://www.autodl.com/docs/proxy_in_instance/

## 1.5 计算性能分析

参考资料：
- https://www.autodl.com/docs/perf/

在终端下，执行以下命令，查看 GPU 使用率：

```sh
nvidia-smi -l 1
```