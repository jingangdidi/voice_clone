use std::path::PathBuf;

use candle_core::{
    Device,
    Tensor,
};

use crate::{
    api::voice::extract_audio_data,
    api::model::tone_color_converter::ToneColorConverter,
    error::MyError,
};

/// default voice
static VOICE_EN_AU:      &str = include_str!("../../extract_base_speaker_data_from_pth/en-au.txt");
static VOICE_EN_BR:      &str = include_str!("../../extract_base_speaker_data_from_pth/en-br.txt");
static VOICE_EN_DEFAULT: &str = include_str!("../../extract_base_speaker_data_from_pth/en-default.txt");
static VOICE_EN_INDIA:   &str = include_str!("../../extract_base_speaker_data_from_pth/en-india.txt");
static VOICE_EN_NEWEST:  &str = include_str!("../../extract_base_speaker_data_from_pth/en-newest.txt");
static VOICE_EN_US:      &str = include_str!("../../extract_base_speaker_data_from_pth/en-us.txt");
static VOICE_ES:         &str = include_str!("../../extract_base_speaker_data_from_pth/es.txt");
static VOICE_FR:         &str = include_str!("../../extract_base_speaker_data_from_pth/fr.txt");
static VOICE_JP:         &str = include_str!("../../extract_base_speaker_data_from_pth/jp.txt");
static VOICE_KR:         &str = include_str!("../../extract_base_speaker_data_from_pth/kr.txt");
static VOICE_ZH:         &str = include_str!("../../extract_base_speaker_data_from_pth/zh.txt");

/// target类型，可以是指定的文件，也可以是内置的默认音色
#[derive(Clone)]
pub enum TargetType {
    EnAu,          // en-au
    EnBr,          // en-br
    EnDefault,     // en-default
    EnIndia,       // en-india
    EnNewest,      // en-newest
    EnUs,          // en-us
    Es,            // es
    Fr,            // fr
    Jp,            // jp
    Kr,            // kr
    Zh,            // zh
    File(PathBuf), // audio file
}

impl TargetType {
    /// 获取target音色Tensor
    pub fn get_target_tensor(&self, tone_color_converter: &ToneColorConverter, device: &Device) -> Result<Tensor, MyError> {
        match self {
            Self::EnAu       => tensor_from_base_speaker(VOICE_EN_AU, device),      // en-au
            Self::EnBr       => tensor_from_base_speaker(VOICE_EN_BR, device),      // en-br
            Self::EnDefault  => tensor_from_base_speaker(VOICE_EN_DEFAULT, device), // en-default
            Self::EnIndia    => tensor_from_base_speaker(VOICE_EN_INDIA, device),   // en-india
            Self::EnNewest   => tensor_from_base_speaker(VOICE_EN_NEWEST, device),  // en-newest
            Self::EnUs       => tensor_from_base_speaker(VOICE_EN_US, device),      // en-us
            Self::Es         => tensor_from_base_speaker(VOICE_ES, device),         // es
            Self::Fr         => tensor_from_base_speaker(VOICE_FR, device),         // fr
            Self::Jp         => tensor_from_base_speaker(VOICE_JP, device),         // jp
            Self::Kr         => tensor_from_base_speaker(VOICE_KR, device),         // kr
            Self::Zh         => tensor_from_base_speaker(VOICE_ZH, device),         // zh
            Self::File(file) => { // audio file
                let target_audio_data = extract_audio_data(&file)?;
                tone_color_converter.extract_se(target_audio_data)
            },
        }
    }
}

/// 将内置base音色转为Tensor，作为target
fn tensor_from_base_speaker(voice: &str, device: &Device) -> Result<Tensor, MyError> {
    let vec_f32 = voice.lines().map(|s| s.parse::<f32>().map_err(|e| MyError::ParseStringError{from: s.to_string(), to: "f32".to_string(), error: e})).collect::<Result<Vec<f32>, MyError>>()?;
    Ok(Tensor::new(vec_f32, device)?.unsqueeze(0)?.unsqueeze(2)?)
}
