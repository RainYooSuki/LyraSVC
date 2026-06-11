# LyraSVC — Singing Voice Conversion

基于 **DiT (Diffusion Transformer) + DDPM** 的歌声转换系统。将源音频的内容、音高、能量特征与目标说话人音色结合，通过扩散模型生成目标 Mel 频谱，再经 HiFiGAN 声码器还原为高质量音频。

## 工作流程

```
训练:
  源音频 → Whisper(PPG) + RMVPE(F0) + nvSTFT(Mel)
       → 条件编码 → RoughMelDecoder(粗mel)
       → DDPM 加噪 → DiT 预测噪声 ε → MSE Loss

推理:
  源音频 → 提取 PPG/F0/Mel
       → Randn 噪声 + 条件 → DPM-Solver++ 20步去噪
       → denorm → HiFiGAN → 音频波形
```

## 项目结构

```
LyraSVC/
├── config/
│   └── config.yaml
├── modules/
│   ├── lyra_model.py            # 主模型 (DiT + DDPM + RoughMelDecoder)
│   ├── dpm_solver.py            # DPM-Solver++ ODE 求解器
│   ├── vocoder.py               # NSF-HiFiGAN 声码器
│   ├── mel.py                   # Mel 谱提取
│   ├── pitch.py                 # F0 提取 (RMVPE)
│   ├── whisper_ppg.py           # Whisper PPG 特征提取
│   ├── nvSTFT.py                # STFT + Mel 底层实现
│   └── rmvpe/                   # RMVPE 模型
├── train.py
├── infer.py
├── preprocess.py
├── diagnose_inference.py
└── README.md
```

## 快速开始

### 0. 准备模型

在项目根目录创建 `Models/` 文件夹，放入预训练模型：

```
Models/
├── whisper-large-v3-turbo/      # Whisper 编码器
├── rmvpe/model.pt                # RMVPE 音高提取
└── pc_nsf_hifigan/               # HiFiGAN 声码器
```

### 1. 准备数据

```
data_raw/
└── <speaker_name>/
    ├── audio_1.wav
    └── audio_2.wav
```

### 2. 预处理

```bash
python preprocess.py
```

结果写入 `data/<speaker>/`。

### 3. 训练

```bash
python train.py

# 中断后恢复
python train.py --resume checkpoints/latest.pt
```

### 4. 推理

```bash
python infer.py \
    --source data_raw/<speaker>/input.wav \
    --output results/output.wav \
    --checkpoint checkpoints/best_ema.pt \
    --speaker 0
```

## 致谢

本项目受益于以下优秀开源工作：

- **[DiffSinger](https://github.com/openvpi/DiffSinger)** — DDPM 扩散框架与 DPM-Solver++ 求解器
- **[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)** — 条件注入模式
- **[ReFlow-VAE-SVC](https://github.com/yxlllc/ReFlow-VAE-SVC)** — nvSTFT Mel 提取实现
- **[DiT (Facebook Research)](https://github.com/facebookresearch/DiT)** — Diffusion Transformer 架构
- **[Whisper (OpenAI)](https://github.com/openai/whisper)** — 内容特征编码
- **[RMVPE](https://github.com/Dream-High/RMVPE)** — 音高提取

## License

MIT
