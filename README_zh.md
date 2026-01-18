# voice_clone
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jingangdidi/voice_clone/blob/main/LICENSE)

[English readme](https://github.com/jingangdidi/voice_clone/blob/main/README.md)

**An [OpenVoice](https://github.com/myshell-ai/OpenVoice)-based voice cloning tool, single executable file (~14M), supporting multiple formats without dependencies on ffmpeg, Python, PyTorch, ONNX.**

**基于[OpenVoice](https://github.com/myshell-ai/OpenVoice)的声音克隆工具，免安装的单个可执行文件（~14M），支持多种格式，不依赖ffmpeg、python、pytorch、onnx**

## 👑 特点
- ​💪​ 单个可执行文件，无需安装
- 🎈 不依赖FFmpeg、Python、PyTorch、ONNX
- 🎨​ 支持多种格式，mp4、mp3、wav等
- 👄 内置多种音色：en-au、en-br、en-default、en-india、en-newest、en-us、es、fr、jp、kr、zh
- 💻​ 支持CPU和GPU

## 🚀 简单使用
**目录结构**
```
你的路径
├─ voice_clone    # 单个可执行文件
└─ checkpoints_v2 # OpenVoice的模型文件
     └─ converter # -m参数指定这个路径，默认./checkpoints_v2/converter
          ├─ config.json
          └─ checkpoint.pth
```
**1. 下载预编译的可执行文件**

[latest release](https://github.com/jingangdidi/voice_clone/releases)

**2. 下载OpenVoice模型参数文件**

[checkpoints_v2](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip)

**3. 命令行运行，将“raw_voice.wav”的音色转换为自己的音色“your_voice.wav”**
```
./voice_clone -s raw_voice.wav -t your_voice.wav
```

## 😁 使用示例
**1. 将原始“test.mp4”的音色转为内置的默认“en-default”**
```
voice_clone -s test.mp4
```
生成1个wav文件:
```
test--en-default.wav # test.mp4 --> en-default
```
**2. 将原始“test1.mp4”和“test2.wav”的音色"转为内置的“zh”**
```
voice_clone -s test1.mp4:test2.wav -t zh
```
生成2个wav文件:
```
test1--zh.wav # test1.mp4 --> zh
test2--zh.wav # test2.wav --> zh
```
**3. 将原始“test.mp4”的音色分别转为内置的“zh”和指定的“my_voice.wav”文件的音色**
```
voice_clone -s test.mp4 -t zh:my_voice.wav
```
生成2个wav文件:
```
test--zh.wav       # test.mp4 --> zh
test--my_voice.wav # test.mp4 --> my_voice.wav
```
**4. 将原始“test1.mp4”和“test2.wav”的音色分别转为内置的“zh”和指定的“my_voice.wav”的音色，并保存提取到的音色数据，用-n指定转换结果文件名，-o指定数据路径**
```
voice_clone -s test1.mp4:test2.wav -t zh:my_voice.wav -S -n result1.wav:result2.wav -o ./result
```
生成5个文件:
```
test1.tone           # test1.mp4的音色数据，下次使用test1.mp4时会跳过提取音色，直接使用test1.tone
test2.tone           # test2.wav的音色数据，下次使用test2.wav时会跳过提取音色，直接使用test2.tone
my_voice.tone        # my_voice.wav的音色数据，下次使用my_voice.wav时会跳过提取音色，直接使用my_voice.tone
./result/result1.wav # test1.mp4 --> zh，指定了输出路径和文件名
./result/result2.wav # test2.wav --> my_voice.wav，指定了输出路径和文件名
```

## ⚡️ 性能
系统: ubuntu 22.04, CPU: i7-13700K, GPU: NVIDIA GeForce RTX 4090, cuda: 12.2
| CPU/GPU | 线程数 | 耗时         | 命令                                    |
| ------- | ------ | ------------ | --------------------------------------- |
| CPU     | 1      | ~40s         | voice_clone -s test_data/test.wav -T 1  |
| CPU     | 10     | ~16s         | voice_clone -s test_data/test.wav -T 10 |
| CPU     | 20     | ~15s         | voice_clone -s test_data/test.wav -T 20 |
| CPU     | all    | ~14s         | voice_clone -s test_data/test.wav -T 0  |
| GPU     |        | ~1.6s        | voice_clone -s test_data/test.wav       |

## 🛠 从源码编译
- **默认使用CPU和不依赖onnx的vad**
```
git clone https://github.com/jingangdidi/voice_clone.git
cd voice_clone
cargo build --release
```

- **使用silero-vad（需要用到onnx）**
```diff
- default = ["simple_vad"]
+ default = ["silero_vad"]
```

- **使用GPU**
```diff
- candle-core = { git = "https://github.com/jingangdidi/candle", package = "candle-core", branch = "main" }
+ candle-core = { git = "https://github.com/jingangdidi/candle", package = "candle-core", branch = "main", features = ["cuda"] }

- candle-nn = { git = "https://github.com/jingangdidi/candle", package = "candle-nn", branch = "main" }
+ candle-nn = { git = "https://github.com/jingangdidi/candle", package = "candle-nn", branch = "main", features = ["cuda"] }
```

## 🚥 命令行参数
```
Usage: voice_clone -s <source> [-t <target>] [-n <name>] [-m <model>] [-S] [-T <thread>] [-o <outpath>]

voice clone

Options:
  -s, --source      source files, colon separated
  -t, --target      target files, colon separated. -t also support base speakers: en-au, en-br, en-default, en-india, en-newest, en-us, es, fr, jp, kr, zh. default: en-default
  -n, --name        result voice file names, colon separated, default: source--target.wav
  -m, --model       openvoice model path, default: ./checkpoints_v2/converter
  -S, --save        save source and target tone color to to the same directory as the specified -s and -t files, maintaining identical nomenclature while altering the format extension to ".tone"
  -T, --thread      cpu threads, 0 means all threads, default: 4
  -o, --outpath     output path, default: ./
  -h, --help        display usage information
```

## 📚 参考
- [OpenVoice](https://github.com/myshell-ai/OpenVoice)
- [candle](https://github.com/huggingface/candle)
- [silero-vad](https://github.com/snakers4/silero-vad)
- [simple-vad](https://github.com/MorenoLaQuatra/vad)
- [stft](https://github.com/phudtran/rustft)
- [audio resample](https://github.com/bmcfee/resampy)

## ⏰ 更新记录
- [2026.01.18] release [v0.1.2](https://github.com/jingangdidi/voice_clone/releases/tag/v0.1.2)
  - 🛠修复: 音频文件数据类型S24和U24解析错误。
  - 🛠修复: 指定`-S`时没有保存`-t`音频文件的`.tone`文件。
- [2025.08.25] release [v0.1.1](https://github.com/jingangdidi/voice_clone/releases/tag/v0.1.1)
  - 🛠修复: 代码中固定的采样率44100替换为声音的实际采样率。
  - ⭐️增加: 支持打印source和target音色的cosine相似度。
  - 💪🏻优化: 支持声音长度小于10秒。
- [2025.08.08] release [v0.1.0](https://github.com/jingangdidi/voice_clone/releases/tag/v0.1.0)
