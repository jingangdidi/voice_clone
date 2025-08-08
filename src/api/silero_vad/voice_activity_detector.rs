use crate::api::silero_vad::{
    silero,
    utils,
    vad_iter,
};

use crate::error::MyError;

/// 调用vad模型，获取音频数据何时说话，何时不说话，返回说话的起始和终止时间在原始数据中的索引位置
/// https://github.com/snakers4/silero-vad
/// 输入的wav音频只支持8khz和16khz，且位深为i16
pub fn split_audio_data_vad(content: Vec<f32>, sr: u32, verbose: bool) -> Result<Vec<(usize, usize)>, MyError> {
    let sample_rate = match sr {
        8000 => utils::SampleRate::EightkHz,
        16000 => utils::SampleRate::SixteenkHz,
        s => return Err(MyError::NormalError{info: format!("unsupported sample rate {}, expect 8 kHz or 16 kHz.", s)})
    };
    if content.is_empty() {
        return Err(MyError::NormalError{info: "audio content is empty.".to_string()})
    }
    let silero = silero::Silero::new(sample_rate)?;
    let vad_params = utils::VadParams {
        sample_rate: sample_rate.into(),
        frame_size: 32,
        min_silence_duration_ms: 1000,
        min_speech_duration_ms: 100,
        speech_pad_ms: 30,
        ..Default::default()
    };
    let mut vad_iterator = vad_iter::VadIter::new(silero, vad_params);
    vad_iterator.process(&content, verbose)?;
    if verbose {
        for timestamp in vad_iterator.speeches() {
            println!("{:?}", timestamp);
        }
    }
    // 返回说话的起始和终止在原始数据中的索引位置
    Ok(vad_iterator.speeches())
}
