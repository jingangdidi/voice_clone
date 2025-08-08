use crate::api::simple_vad::energy_vad::EnergyVAD;

use crate::error::MyError;

/// 调用vad模型，获取音频数据何时说话，何时不说话，返回说话的起始和终止时间在原始数据中的索引位置
/// https://github.com/MorenoLaQuatra/vad
pub fn split_audio_data_vad(content: Vec<f32>, sr: u32, verbose: bool) -> Result<Vec<(usize, usize)>, MyError> {
    if content.is_empty() {
        return Err(MyError::NormalError{info: "audio content is empty.".to_string()})
    }
    let vad = EnergyVAD::new(sr, 25, 20, 0.05, 0.95);
    let result = vad.apply_vad(&content); // get the audio file with only speech
    if verbose {
        for timestamp in &result {
            println!("{:?}", timestamp);
        }
    }
    // 返回说话的起始和终止在原始数据中的索引位置
    Ok(result)
}
