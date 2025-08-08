use candle_nn::{
    VarBuilder,
};

use crate::{
    api::model::synthesizer::SynthesizerTrn,
    config::HParams,
    error::MyError,
};

/// 内部封装的模型
pub struct OpenVoiceBaseClass {
    pub model:  SynthesizerTrn, // 语音模型
    //device: Device,         // cup, cuda
}

impl OpenVoiceBaseClass {
    /// 初始化模型
    pub fn new(hps: &HParams, vb: VarBuilder) -> Result<Self, MyError> {
        // 初始化语音模型
        //let device = vb.device().clone();
        let model = SynthesizerTrn::new(hps, vb)?;
        Ok(OpenVoiceBaseClass{
            model,
            //device，
        })
    }
}
