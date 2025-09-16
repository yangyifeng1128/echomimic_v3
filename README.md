# EchoMimic V3: 13亿参数即可实现统一多模态、多任务人体动画生成

## &#x1F4E3; 更新日志

* [2025.08.12] 🔥 **仅需12G显存生成视频**，量化版本 [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) 发布。 查看 [教程](https://www.bilibili.com/video/BV1W8tdzEEVN)。感谢 @[gluttony-10](https://github.com/gluttony-10) 贡献。
* [2025.08.12] 🔥 EchoMimic V3 支持 16GB 显存，使用 [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic)。感谢 @[smthemex](https://github.com/smthemex) 的贡献。
* [2025.08.10] 🔥 [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app.py) 已发布，感谢 @[gluttony-10](https://github.com/gluttony-10) 的贡献。
* [2025.08.09] 🔥 我们在 ModelScope 上发布了 [模型](https://modelscope.cn/models/BadToBest/EchoMimicV3)。
* [2025.08.08] 🔥 我们在 Huggingface 上发布了 [代码](https://github.com/antgroup/echomimic_v3) 和 [模型](https://huggingface.co/BadToBest/EchoMimicV3)。
* [2025.07.08] 🔥 我们的 [论文](https://arxiv.org/abs/2507.03905) 在 arxiv 上公开。

## 快速开始

### 环境配置
- 测试系统环境：Ubuntu 22.04, Cuda >= 12.8
- 测试 GPU：RTX 5090 (32GB)
- 测试 Python：3.12

### 🛠️ 安装

#### 1. 安装依赖包

```sh
uv sync
```

#### 2. 准备模型

| 模型名称            | 下载链接                                                                     | 备注              |
| ------------------- | ---------------------------------------------------------------------------- | ----------------- |
| Wan2.1-Fun-1.3B-InP | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | 基础模型          |
| wav2vec2-base       | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | 音频编码器        |
| EchoMimicV3-preview | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                | EchoMimic V3 权重 |

在终端下，执行以下命令，下载模型：

```sh
uv run hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP
uv run hf download facebook/wav2vec2-base-960h
uv run hf download BadToBest/EchoMimicV3
```

将下载完成的模型文件，按照以下目录结构进行组织：

```sh
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
``` 

```sh
mkdir models
cd models

ln -s /root/autodl-tmp/hf/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP Wan2.1-Fun-V1.1-1.3B-InP
ln -s /root/autodl-tmp/hf/hub/models--facebook--wav2vec2-base-960h wav2vec2-base-960h
ln -s /root/autodl-tmp/hf/hub/diffusion_pytorch_model.safetensors diffusion_pytorch_model.safetensors
```

### 🔑 快速推理

```
python app_mm.py
```

#### 提示
> - 音频 CFG：音频 CFG `audio_guidance_scale` 最佳范围为 2~3。增加音频 CFG 值可以改善唇同步效果，减少音频 CFG 值可以提高视觉质量。
> - 文本 CFG：文本 CFG `guidance_scale` 最佳范围为 3~6。增加文本 CFG 值可以更好地遵循提示词，减少文本 CFG 值可以提高视觉质量。
> - TeaCache：`teacache_threshold` 的最佳范围为 0~0.1。
> - 采样步数：头部动画为 5 步，全身动作为 15~25 步。
> - ​长视频生成：如果需要生成超过 138 帧的视频，可以使用长视频 CFG。
> - 尝试降低`partial_video_length`节省显存。
