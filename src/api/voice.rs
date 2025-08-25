use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::SystemTime;

use candle_core::{
    Device,
    DType,
    Tensor,
    utils::cuda_is_available,
};
use candle_nn::VarBuilder;
use memmap2::MmapOptions;
use safetensors::SafeTensors;

use crate::{
    api::audio::{
        load_audio::load_audio,
        resample::resample,
    },
    api::model::tone_color_converter::ToneColorConverter,
    error::MyError,
    parse_paras::Voice,
};

#[cfg(feature = "silero_vad")]
use crate::api::silero_vad::voice_activity_detector::split_audio_data_vad;
#[cfg(feature = "simple_vad")]
use crate::api::simple_vad::voice_activity_detector::split_audio_data_vad;


/// 将source语音的音色换为target（自己准备的）语音的音色，内容不变
/// 生成的语音内容与source相同，只是音色变为target
pub fn convert_voice(voices: Vec<Voice>, config: &Path, ckpt: &Path, save: bool) -> Result<(), MyError> {
    // 加载参数文件
    let device = if cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    let vb = VarBuilder::from_pth(ckpt, DType::F32, &device)?;

    // 初始化模型网络
    let tone_color_converter = ToneColorConverter::new(config, vb.clone())?;

    // 存储每个文件的音色，这样就不必重复提取相同文件的音色
    let mut tone_colors: HashMap<String, Tensor> = HashMap::new();

    // 遍历每对source和target语音对，进行音色转换
    for v in voices {
        let t1 = SystemTime::now();
        if !tone_colors.contains_key(&v.src_path) {
            let source_se = if v.tone_s.exists() && v.tone_s.is_file() { // 直接读取source已经提取的音色文件
                load_tone_color_from_file(&v.tone_s, &device)?
            } else { // 提取source音色
                let source_audio_data = extract_audio_data(&v.source)?;
                tone_color_converter.extract_se(source_audio_data)?
            };
            //println!("\n\nsource_se:\n{:?}", source_se.squeeze(0)?.squeeze(1)?.to_vec1::<f32>());
            if save {
                source_se.save_safetensors("tone", &v.tone_s)?;
            }
            // 把新提取的音色插入到HashMap中
            tone_colors.insert(v.src_path.clone(), source_se);
        }

        if !tone_colors.contains_key(&v.tgt_path) {
            let target_se = match v.tone_t {
                Some(tone) => if tone.exists() && tone.is_file() { // 直接读取target已经提取的音色文件
                    let target_se = load_tone_color_from_file(&tone, &device)?;
                    if save {
                        target_se.save_safetensors("tone", &tone)?;
                    }
                    target_se
                } else {
                    v.target.get_target_tensor(&tone_color_converter, &device)?
                },
                None => v.target.get_target_tensor(&tone_color_converter, &device)?,
            };
            //println!("\n\ntarget_se: {:?}\n{:?}", target_se.dims(), target_se.squeeze(0).unwrap().squeeze(1).unwrap().to_vec1::<f32>());
            // 把新提取的音色插入到HashMap中
            tone_colors.insert(v.tgt_path.clone(), target_se);
        }

        let source_se = tone_colors.get(&v.src_path).unwrap();
        let target_se = tone_colors.get(&v.tgt_path).unwrap();

        // source and target tone cosine similarity
        let similarity = cosine_similarity(source_se, target_se)?;

        // 生成语音
        tone_color_converter.convert(
            &v.source, // audio_src_path
            source_se, // src_se
            target_se, // tgt_se
            0.3, // tau
            &v.out_file, // 输出文件
        )?;
        // 打印耗时
        let t2 = SystemTime::now();
        println!("convert tone color: {} --> {} (elapsed time: {}, cosine similarity: {})", v.src_name, v.tgt_name, elapsed_time(t1, t2), similarity);
    }
    Ok(())
}

/// 计算耗时
pub fn elapsed_time(start: SystemTime, end: SystemTime) -> String {
    let nano = end.duration_since(start).unwrap().as_nanos();
    if nano > 1_000_000_000 {
        format!("{:.2}s", (nano / 1_000_000_000) as f32 + (nano % 1_000_000_000) as f32 / 1e9)
    } else if nano > 1_000_000 {
        format!("{:.2}ms", (nano / 1_000_000) as f32 + (nano % 1_000_000) as f32 / 1e6)
    } else if nano > 1_000 {
        format!("{:.2}us", (nano / 1_000) as f32 + (nano % 1_000) as f32 / 1e3)
    } else {
        format!("{:.2}ns", nano as f32)
    }
}

/// Computes the cosine similarity between two tone Tensor
/// https://en.wikipedia.org/wiki/Cosine_similarity
/// https://github.com/gaspiman/cosine_similarity/blob/master/cosine.go
fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32, MyError> {
    let vec_a = a.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;
    let vec_b = b.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;

    let mut ab: f32 = 0.0;
    let mut sum_a: f32 = 0.0;
    let mut sum_b: f32 = 0.0;
    for (i, j) in vec_a.iter().zip(vec_b.iter()) {
        ab += i * j;
        sum_a += i.powf(2.0);
        sum_b += j.powf(2.0);
    }

    Ok(ab / (sum_a.sqrt() * sum_b.sqrt()))
}

/// 1. 读取音频文件，重采样
/// 2. 使用vad获取activity的起始、终止位置索引
/// 3. 对原始f32的音频数据进行拆分
pub fn extract_audio_data(file: &Path) -> Result<Vec<Vec<f32>>, MyError> {
    // 1. 读取音频文件，重采样
    /*
    // 用ffmpeg重采样
    use std::process::Command;

    let mut cmd = Command::new("ffmpeg");
    cmd.args(&[
        "-nostdin",
        "-threads", "0",
        "-i", file.to_str().unwrap(),
        "-f", "s16le",
        "-ac", "1", // 只获取1个通道
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-",
    ]);

    // 执行命令并捕获输出
    let output = cmd.output().expect("Failed to execute ffmpeg");

    // 检查命令执行是否成功
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("FFmpeg failed: {}", stderr);
    }

    let bytes = &output.stdout;
    let len = bytes.len();

    // 确保字节长度为偶数（每个 i16 占 2 字节）
    assert_eq!(len % 2, 0);

    // 将字节解析为 i16 数组（小端格式）
    let audio_data_i16: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    // 转换为 float32 并归一化
    let float_samples: Vec<f32> = audio_data
        .into_iter()
        .map(|s| s as f32 / 32768.0)
        .collect();

    // Cheap normalization of the volume
    // https://github.com/linto-ai/whisper-timestamped/blob/master/whisper_timestamped/transcribe.py
    // audio = audio / max(0.1, audio.abs().max())
    let audio_data_f32: Vec<f32> = audio_data_i16.iter().map(|s| *s as f32 / 32768.0).collect();
    */

    // 这样就不需要调用ffmpeg了！
    let audio_data_f32 = load_audio(file, Some(16000))?.samples; // 这是f32的音频数据，44100 --> 16000

    let for_norm = f32::max(audio_data_f32.iter().map(|x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(), 0.1);
    let audio_data_f32_norm: Vec<f32> = audio_data_f32.iter().map(|x| x / for_norm).collect();

    // 2. 使用vad获取activity的起始、终止位置索引
    let segments: Vec<(usize, usize)> = split_audio_data_vad(audio_data_f32_norm, 16000, false)?;

    // 3. 对f32的音频数据进行拆分
    let audio_data = load_audio(file, None)?;
    let audio_data_f32 = audio_data.samples; // 这是f32的音频数据
    let sample_rate: u32 = audio_data.sample_rate;

    let ratio = sample_rate as f32 / 16000.0;
    let mut final_audio_data: Vec<Vec<f32>> = vec![];
    let mut final_audio_data_merged: Vec<f32> = vec![];
    for i in segments {
        //println!("start: {}, end: {}", i.0, i.1);
        let tmp_data = audio_data_f32[(i.0 as f32 * ratio).round() as usize..(i.1 as f32 * ratio).round() as usize].to_vec();
        final_audio_data_merged.extend_from_slice(&tmp_data);
        final_audio_data.push(tmp_data);
    }
    let total_length = final_audio_data_merged.len();
    let num_splits = if total_length > (sample_rate as usize * 10) {
        (total_length as f32 / 10.0 / sample_rate as f32).round() as usize
    } else {
        1
    }; // 默认按照10秒拆分
    let interval = total_length / num_splits; // 每段多长

    let mut final_audio_data: Vec<Vec<f32>> = vec![];
    let mut start = 0;
    let mut end: usize;
    for i in 0..num_splits {
        end = total_length.min(start + interval);
        if i == num_splits - 1 {
            end = total_length;
        }
        let y = &final_audio_data_merged[start..end];
        let n_samples = (y.len() as f32 * 22050.0 / sample_rate as f32).ceil() as usize;
        let mut after_resample = resample(y, sample_rate, 22050, true);
        if after_resample.len() > n_samples { // 截取前n_samples个
            after_resample.truncate(n_samples);
        } else { // 在最后补0
            after_resample.extend(vec![0.0; n_samples-after_resample.len()]);
        }
        final_audio_data.push(after_resample);
        start = end;
    }

    Ok(final_audio_data)
}

/// 从音色文件（以`.tone`为格式后缀，safetensors格式）中读取已经提取的音色Tensor
fn load_tone_color_from_file(file: &Path, device: &Device) -> Result<Tensor, MyError> {
    let tone_file = File::open(file).map_err(|e| MyError::OpenFileError{file: file.to_str().unwrap().to_string(), error: e})?;
    let buffer = unsafe { MmapOptions::new().map(&tone_file)? };
    let safe_tensors = SafeTensors::deserialize(&buffer)?;
    let tensor_view = safe_tensors.tensor("tone")?;
    Ok(Tensor::from_raw_buffer(tensor_view.data(), DType::F32, tensor_view.shape(), device)?)
}
