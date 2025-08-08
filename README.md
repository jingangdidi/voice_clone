# voice_clone
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jingangdidi/voice_clone/blob/main/LICENSE)

[中文文档](https://github.com/jingangdidi/voice_clone/blob/main/README_zh.md)

**An [OpenVoice](https://github.com/myshell-ai/OpenVoice)-based voice cloning tool, single executable file (~14M), supporting multiple formats without dependencies on ffmpeg, Python, PyTorch, ONNX.**

**基于[OpenVoice](https://github.com/myshell-ai/OpenVoice)的声音克隆工具，免安装的单个可执行文件（~14M），支持多种格式，不依赖ffmpeg、python、pytorch、onnx**

## 👑 Features
- ​💪​ Single-file executable - no installation required
- 🎈 Independent of FFmpeg, Python, PyTorch, and ONNX
- 🎨​ Support multiple formats (e.g. mp4, mp3, wav)
- 👄 Offer multiple built-in base speakers: en-au, en-br, en-default, en-india, en-newest, en-us, es, fr, jp, kr, zh
- 💻​ Support CPU & GPU

## 🚀 Quick-Start
**structure**
```
some dir
├─ voice_clone    # single executable file
└─ checkpoints_v2 # OpenVoice model
     └─ converter # use -m specify this dir, default: ./checkpoints_v2/converter
          ├─ config.json
          └─ checkpoint.pth
```
**1. download a pre-built binary**

[latest release](https://github.com/jingangdidi/voice_clone/releases)

**2. download OpenVoice modle**

[checkpoints_v2](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip)

**3. convert some voice to your voice**
```
./voice_clone -s raw_voice.wav -t your_voice.wav
```

## 😁 Usage Example
**1. convert tone color of "test.mp4" to default built-in base speaker "en-default"**
```
voice_clone -s test.mp4
```
output 1 file:
```
test--en-default.wav # test.mp4 --> en-default
```
**2. convert tone color of "test1.mp4" and "test2.wav" to built-in base speaker "zh"**
```
voice_clone -s test1.mp4:test2.wav -t zh
```
output 2 files:
```
test1--zh.wav # test1.mp4 --> zh
test2--zh.wav # test2.wav --> zh
```
**3. convert tone color of "test.mp4" to built-in base speaker "zh" and "my_voice.wav"**
```
voice_clone -s test.mp4 -t zh:my_voice.wav
```
output 2 files:
```
test--zh.wav       # test.mp4 --> zh
test--my_voice.wav # test.mp4 --> my_voice.wav
```
**4. convert tone color of "test1.mp4" and "test2.wav" to built-in base speaker "zh" and "my_voice.wav", save extracted tone color**
```
voice_clone -s test1.mp4:test2.wav -t zh:my_voice.wav -S -n result1.wav:result2.wav -o ./result
```
output 5 files:
```
test1.tone           # test1.mp4 tone color, next time use test1.mp4, will skip extract tone color from test1.mp4, use test1.tone directly
test2.tone           # test2.wav tone color, next time use test2.wav, will skip extract tone color from test2.wav, use test2.tone directly
my_voice.tone        # my_voice.wav tone color, next time use my_voice.wav, will skip extract tone color from my_voice.wav, use my_voice.tone directly
./result/result1.wav # test1.mp4 --> zh
./result/result2.wav # test2.wav --> my_voice.wav
```

## ⚡️ Performance
os: ubuntu 22.04, CPU: i7-13700K, GPU: NVIDIA GeForce RTX 4090, cuda: 12.2
| CPU/GPU | thread | elapsed time | command                                 |
| ------- | ------ | ------------ | --------------------------------------- |
| CPU     | 1      | ~40s         | voice_clone -s test_data/test.wav -T 1  |
| CPU     | 10     | ~16s         | voice_clone -s test_data/test.wav -T 4  |
| CPU     | 20     | ~15s         | voice_clone -s test_data/test.wav -T 10 |
| CPU     | all    | ~14s         | voice_clone -s test_data/test.wav -T 0  |
| GPU     |        | ~1.6s        | voice_clone -s test_data/test.wav       |

## 🛠 Building from source
- **default use cpu and simple vad (not require onnx)**
```
git clone https://github.com/jingangdidi/voice_clone.git
cd voice_clone
cargo build --release
```

- **use silero vad (require onnx)**
```diff
- default = ["simple_vad"]
+ default = ["silero_vad"]
```

- **use GPU**
```diff
- candle-core = { git = "https://github.com/jingangdidi/candle", package = "candle-core", branch = "main" }
+ candle-core = { git = "https://github.com/jingangdidi/candle", package = "candle-core", branch = "main", features = ["cuda"] }

- candle-nn = { git = "https://github.com/jingangdidi/candle", package = "candle-nn", branch = "main" }
+ candle-nn = { git = "https://github.com/jingangdidi/candle", package = "candle-nn", branch = "main", features = ["cuda"] }
```

## 🚥 Arguments
```
Usage: voice_clone.exe -s <source> [-t <target>] [-n <name>] [-m <model>] [-S] [-T <thread>] [-o <outpath>]

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

## 📚 Acknowledgements
- [OpenVoice](https://github.com/myshell-ai/OpenVoice)
- [candle](https://github.com/huggingface/candle)
- [silero-vad](https://github.com/snakers4/silero-vad)
- [simple-vad](https://github.com/MorenoLaQuatra/vad)
- [stft](https://github.com/phudtran/rustft)
- [audio resample](https://github.com/bmcfee/resampy)

## ⏰ changelog
- [2025.08.06] release v0.1.0
