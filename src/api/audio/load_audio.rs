use std::fs::File;
use std::path::Path;

use hound::{
    SampleFormat,
    WavSpec,
    WavWriter,
};
use symphonia::core::{
    audio::{
        AudioBufferRef,
        Signal,
    },
    codecs::{
        DecoderOptions,
        CODEC_TYPE_NULL,
    },
    errors::Error as symphonia_error,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use crate::{
    api::audio::resample::resample,
    error::MyError,
};

// 实现python的librosa读取、保存音频文件
// 示例1：直接调用load_audio函数读取音频文件（类似`librosa.load`）
// +-------------------------------------------------------------------------------------------+
// | let audio_data = load_audio("audio.wav", Some(22050))?;                                   | 设为None则不重采样，保持原始采样率
// | println!("Loaded {} samples at {} Hz", audio_data.samples.len(), audio_data.sample_rate); |
// +-------------------------------------------------------------------------------------------+
// 示例2：创建Audio结构体导入音频文件（更灵活）
// +--------------------------------------------------------------------------------------------------------------------------------+
// | let loader = Audio::new().with_sample_rate(16000);                                                                             |
// | let audio_data = loader.load("audio.mp3")?;                                                                                    |
// | println!("Audio data: {} samples, {} Hz, {} channels", audio_data.samples.len(), audio_data.sample_rate, audio_data.channels); |
// +--------------------------------------------------------------------------------------------------------------------------------+
// 示例3：直接调用load_audio函数读取音频文件，再调用save_audio函数保存为指定位深的wav文件
// +------------------------------------------------------------------------------------------------+
// | let audio_data = load_audio("audio.wav", Some(16000))?;                                        | 设为None则不重采样，保持原始采样率
// | println!("Loaded {} samples at {} Hz", audio_data.samples.len(), audio_data.sample_rate);      |
// | save_audio(&audio_data.samples, audio_data.sample_rate, 16, Path::new("audio_16000_16.wav"))?; |
// +------------------------------------------------------------------------------------------------+

/// 存储音频数据
pub struct AudioData {
    pub samples:     Vec<f32>,
    pub sample_rate: u32,
    pub channels:    u16,
}

/// 返回不同类型的音频数据
pub enum DataType {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I2432(Vec<i32>),
    F32(Vec<f32>),
}

/// 音频文件读取器
pub struct Audio {
    default_sample_rate: Option<u32>, // None表示使用原始采样率，不需要重采样，Some表示指定了新采样率，需要重采样
}

impl Audio {
    /// 初始化读取器
    pub fn new() -> Self {
        Self{default_sample_rate: None}
    }

    /// 设置采样率
    pub fn with_sample_rate(mut self, sr: u32) -> Self {
        self.default_sample_rate = Some(sr);
        self
    }

    /// 加载音频文件，类似于python的librosa.load()
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<AudioData, MyError> {
        // 读取音频文件
        let file = File::open(&path).map_err(|e| MyError::OpenFileError{file: format!("{:?}", path.as_ref().display()), error: e})?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // 创建格式读取器
        let hint = Hint::new(); // A Hint provides additional information and context when probing a media source stream.
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let decoder_opts = DecoderOptions::default();

        // Probe the media source
        let probed = symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

        // Get the instantiated format reader
        let mut format = probed.format;

        // 找到第一个音频轨道
        let track = format.tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL).unwrap();

        // Store the track identifier, it will be used to filter packets
        let track_id = track.id;

        // 创建解码器
        let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

        // 存储解码的数据
        let mut audio_samples = Vec::new();
        let original_sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
        let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1) as u16;
        //println!("audio_samples length: {}, original_sample_rate: {}, self.default_sample_rate: {:?}", audio_samples.len(), original_sample_rate, self.default_sample_rate);

        // 解码音频数据
        loop {
            // Get the next packet from the media format
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia_error::ResetRequired) => {
                    // 重置解码器并继续
                    decoder.reset();
                    continue
                }
                Err(symphonia_error::IoError(e)) 
                    if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break
                }
                Err(err) => return Err(err)?,
            };

            // If the packet does not belong to the selected track, skip over it
            if packet.track_id() != track_id {
                continue
            }

            // Decode the packet into audio samples
            match decoder.decode(&packet)? {
                AudioBufferRef::U8(buf) => { // u8类型，值范围[0, 2^8-1]，即[0, 255]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] as f32 + ch1[i] as f32) / 2.0;
                            audio_samples.push((avg - 128.0) / 128.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push((sample as f32 - 128.0) / 128.0);
                        }
                    }
                },
                AudioBufferRef::U16(buf) => { // u6类型，值范围[0, 2^16-1]，即[0, 65535]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] as f32 + ch1[i] as f32) / 2.0;
                            audio_samples.push((avg - 32768.0) / 32768.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push((sample as f32 - 32768.0) / 32768.0);
                        }
                    }
                },
                AudioBufferRef::U24(buf) => { // u24类型不能直接转f32，U24内部是u32，先获取内部的u32值，值范围[0, 2^32-1]，即[0, 4294967295]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i].inner() as f32 + ch1[i].inner() as f32) / 2.0;
                            audio_samples.push((avg - 2147483648.0) / 2147483648.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push((sample.inner() as f32 - 2147483648.0) / 2147483648.0); // 不能直接转f32，U24内部是u32，先通过`.inner()`方法获取内部u32，再转为f32
                        }
                    }
                },
                AudioBufferRef::U32(buf) => { // u32类型，值范围[0, 2^32-1]，即[0, 4294967295]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] as f32 + ch1[i] as f32) / 2.0;
                            audio_samples.push((avg - 2147483648.0) / 2147483648.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push((sample as f32 - 2147483648.0) / 2147483648.0);
                        }
                    }
                },
                AudioBufferRef::S8(buf) => { // i8类型，值范围[-(2^7), 2^7-1]，即[-128, 127]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] as f32 + ch1[i] as f32) / 2.0;
                            audio_samples.push(avg / 128.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push(sample as f32 / 128.0);
                        }
                    }
                },
                AudioBufferRef::S16(buf) => { // i16类型，值范围[-(2^15), 2^15-1]，即[-32768, 32767]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] as f32 + ch1[i] as f32) / 2.0;
                            audio_samples.push(avg / 32768.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push(sample as f32 / 32768.0);
                        }
                    }
                },
                AudioBufferRef::S24(buf) => { // i24类型不能直接转f32，i24内部是i32，先获取内部的i32值，值范围[-(2^31), 2^31-1]，即[-2147483648, 2147483647]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i].inner() as f32 + ch1[i].inner() as f32) / 2.0;
                            audio_samples.push(avg / 2147483648.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push(sample.inner() as f32 / 2147483648.0); // 不能直接转f32，i24内部是i32，先通过`.inner()`方法获取内部i32，再转为f32
                        }
                    }
                },
                AudioBufferRef::S32(buf) => { // i32类型，值范围[-(2^31), 2^31-1]，即[-2147483648, 2147483647]，转为f32类型，值范围标准化到[-1, 1]
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] as f32 + ch1[i] as f32) / 2.0;
                            audio_samples.push(avg / 2147483648.0);
                        }
                    } else {
                        for &sample in buf.chan(0) {
                            audio_samples.push(sample as f32 / 2147483648.0);
                        }
                    }
                },
                AudioBufferRef::F32(buf) => { // f32类型，本身已经是[-1, 1]范围，不需要转换和标准化
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] + ch1[i]) / 2.0;
                            audio_samples.push(avg);
                        }
                    } else {
                        audio_samples.extend_from_slice(buf.chan(0));
                    }
                },
                AudioBufferRef::F64(buf) => { // f64类型，需要转为f32类型，且可能超出[-1, 1]范围，因此需要检查下，如果最大/最小值超出[-1, 1]范围，则用最大/最小值标准化到[-1, 1]范围
                    if channels == 2 {
                        let ch0 = buf.chan(0);
                        let ch1 = buf.chan(1);
                        let mut avg_vec: Vec<f32> = Vec::with_capacity(ch0.len());
                        let mut max_abs_value = 0.0;
                        for i in 0..ch0.len() {
                            let avg = (ch0[i] + ch1[i]) as f32 / 2.0;
                            if avg > max_abs_value {
                                max_abs_value = avg;
                            }
                            avg_vec.push(avg);
                        }
                        if max_abs_value > 1.0 { // 需要进行标准化
                            avg_vec = avg_vec.iter().map(|s| s / max_abs_value).collect();
                        }
                        audio_samples.extend_from_slice(&avg_vec);
                    } else {
                        let max_abs_value = buf.chan(0).iter().fold(0.0_f64, |max, &x| max.max(x.abs())) as f32; // 获取原始f64数据绝对值的最大值
                        if max_abs_value > 1.0 { // 需要进行标准化
                            for &sample in buf.chan(0) {
                                audio_samples.push(sample as f32 / max_abs_value);
                            }
                        } else { // 不需要进行标准化
                            for &sample in buf.chan(0) {
                                audio_samples.push(sample as f32);
                            }
                        }
                    }
                },
            }
        }

        // 处理重采样
        let final_samples = if let Some(target_sr) = self.default_sample_rate {
            if target_sr != original_sample_rate {
                resample(&audio_samples, original_sample_rate, target_sr, true)
            } else {
                audio_samples
            }
        } else {
            audio_samples
        };

        Ok(AudioData{
            samples: final_samples,
            sample_rate: self.default_sample_rate.unwrap_or(original_sample_rate),
            channels,
        })
    }

    /// 将音频数据转为指定位深，返回数据
    pub fn convert_audio_depth(
        &self,
        audio_data: &AudioData, // 音频数据
        bits_per_sample: u16, // 位深，例如：8（表示i8）、16（表示i16）、24（表示i24）、32（表示i32）、320（表示f32）
    ) -> Result<DataType, MyError> {
        // 根据位深度转换样本，hound只支持：i8、i16、i32、f32
        match bits_per_sample {
            8 => Ok(DataType::I8(audio_data.samples.iter().map(|s| (s * 128.0) as i8).collect())), // i8
            16 => Ok(DataType::I16(audio_data.samples.iter().map(|s| (s * 32768.0) as i16).collect())), // i16
            24 | 32 => Ok(DataType::I2432(audio_data.samples.iter().map(|s| (s * 2147483648.0) as i32).collect())), // i24和i32
            320 => Ok(DataType::F32(audio_data.samples.clone())), // f32，暂时先用320表示f32，与i32进行区分
            _ => Err(MyError::NormalError{info: format!("unsupported bit depth: {}", bits_per_sample)}),
        }
    }

    /// 保存音频数据为WAV文件，支持自定义位深
    pub fn save_wav_with_depth<P: AsRef<Path>>(
        &self,
        audio_data: &AudioData, // 音频数据
        bits_per_sample: u16, // 位深，例如：8（表示i8）、16（表示i16）、24（表示i24）、32（表示i32）、320（表示f32）
        output_file: P, // 输出文件
    ) -> Result<(), MyError> {
        // wav对象
        let spec = WavSpec {
            channels: audio_data.channels,
            sample_rate: audio_data.sample_rate,
            bits_per_sample: if bits_per_sample == 320 {
                32
            } else {
                bits_per_sample
            },
            sample_format: if bits_per_sample == 320 {
                SampleFormat::Float
            } else {
                SampleFormat::Int
            },
        };

        // 保存数据的writer
        let mut writer = WavWriter::create(&output_file, spec)?;

        // 根据位深度转换样本，hound只支持：i8、i16、i32、f32
        match self.convert_audio_depth(&audio_data, bits_per_sample)? {
            DataType::I8(data) => {
                for d in data {
                    writer.write_sample(d)?;
                }
            },
            DataType::I16(data) => {
                for d in data {
                    writer.write_sample(d)?;
                }
            },
            DataType::I2432(data) => {
                for d in data {
                    writer.write_sample(d)?;
                }
            },
            DataType::F32(data) => {
                for d in data {
                    writer.write_sample(d)?;
                }
            },
        }

        Ok(writer.finalize()?)
    }
}

/// 便捷函数，模拟`librosa.load`接口
pub fn load_audio<P: AsRef<Path>>(audio_path: P, sr: Option<u32>) -> Result<AudioData, MyError> {
    let mut loader = Audio::new();
    if let Some(sample_rate) = sr {
        loader = loader.with_sample_rate(sample_rate);
    }
    loader.load(audio_path)
}

/// 便捷函数，保存音频为指定位深的wav文件
/// 这个函数输入的是已经读取的音频数据，也就是需要先使用`load_audio`函数读取音频文件，再调用该函数
pub fn save_audio<P: AsRef<Path>>(samples: &[f32], sample_rate: u32, bits_per_sample: u16, output_file: P) -> Result<(), MyError> {
    let audio_data = AudioData {
        samples: samples.to_vec(),
        sample_rate,
        channels: 1, // 默认单声道
    };
    let loader = Audio::new();
    loader.save_wav_with_depth(&audio_data, bits_per_sample, output_file)
}
