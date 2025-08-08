use std::path::Path;

use candle_core::{
    D,
    Device,
    Tensor,
};
use candle_nn::{
    VarBuilder,
};

use crate::{
    api::audio::load_audio::{load_audio, save_audio},
    api::model::open_voice_base::OpenVoiceBaseClass,
    api::stft::{
        Stft,
        WindowFunction,
    },
    config::HParams,
    error::MyError,
};

/// 提取语音音色
pub struct ToneColorConverter {
    model:  OpenVoiceBaseClass,
    device: Device,
    hps:    HParams,
}

impl ToneColorConverter {
    pub fn new(config: &Path, vb: VarBuilder) -> Result<Self, MyError> {
        let hps = HParams::new(config)?; // 读取config.json
        let device = vb.device().clone();
        Ok(ToneColorConverter {
            model: OpenVoiceBaseClass::new(&hps, vb)?,
            device,
            hps,
        })
    }

    /// 提取音色
    pub fn extract_se(&self, ref_wav_list: Vec<Vec<f32>>) -> Result<Tensor, MyError> {
        let mut gs: Vec<Tensor> = Vec::with_capacity(ref_wav_list.len());

        let stft = Stft::new(
            self.hps.data.filter_length, // 1024
            self.hps.data.hop_length, // 256
            WindowFunction::Hann,
            true,
        );

        // 遍历每个切分的数据进行stft计算
        for data in ref_wav_list {
            let spectrogram = stft.forward(data)?;
            //println!("complex2: {:?}", spectrogram);
            let mut spectrogram = Tensor::new(spectrogram, &self.device)?; // [513, 976]
            spectrogram = spectrogram.unsqueeze(0)?; // [1, 513, 976]
            //println!("complex2 tensor: {:?}, {:?}", spectrogram.dims(), spectrogram.squeeze(0)?.to_vec2::<f32>());
            //let spectrogram2 = spectrogram.transpose(1, 2)?.squeeze(0)?.to_vec2::<f32>();
            //println!("complex22 tensor: {:?}", spectrogram2);
            //spectrogram = self.model.model.ref_enc.forward(&spectrogram.unsqueeze(0)?.transpose(1, 2)?, None)?.unsqueeze(D::Minus1)?; // test.wav: [1, 256, 1], [1, 256, 1], srx.wav: [1, 256, 1]
            // 传递给`ref_enc.forward`的shape是`[1, 976, 513]`
            spectrogram = self.model.model.ref_enc.forward(&spectrogram.transpose(1, 2)?)?.unsqueeze(D::Minus1)?; // test.wav: [1, 256, 1], [1, 256, 1], srx.wav: [1, 256, 1]
            //println!("ref_enc: {:?}, {:?}", spectrogram.dims(), spectrogram.squeeze(0)?.squeeze(1)?.to_vec1::<f32>());

            gs.push(spectrogram);
        }
        Ok(Tensor::stack(&gs, 0)?.mean(0)?) // [1, 256, 1]
    }

    /// 音色转换
    pub fn convert(
        &self,
        audio_src_path: &Path,
        src_se: &Tensor,
        tgt_se: &Tensor,
        tau: f32,
        outfile: &Path,
    ) -> Result<(), MyError> {
        // 读取source音频文件，得到Vec<f32>
        let audio_data_f32 = load_audio(audio_src_path, Some(22050))?.samples; // 这是f32的音频数据，44100 --> 22050

        let stft = Stft::new(
            self.hps.data.filter_length, // 1024
            self.hps.data.hop_length, // 256
            WindowFunction::Hann,
            true,
        );

        let spectrogram = stft.forward(audio_data_f32)?;
        //println!("convert: {:?}", spectrogram);
        let spectrogram = Tensor::new(spectrogram, &self.device)?.unsqueeze(0)?; // [1, 513, 2078]
        //println!("convert tensor: {:?}, {:?}", spectrogram.dims(), spectrogram.squeeze(0)?.to_vec2::<f32>());

        // 长度
        let shape = spectrogram.dims();
        let spec_lengths = shape[shape.len()-1]; // [1]
        //let spec_lengths = 1;

        // 音色转换
        //println!("spectrogram: {:?}", spectrogram);
        let audio_data = self
            .model
            .model
            .voice_conversion(spectrogram, spec_lengths, src_se, tgt_se, tau)? // 531968
            .0
            .squeeze(0)?
            .squeeze(0)?
            .to_vec1::<f32>()?;

        save_audio(&audio_data, 22050, 320, outfile)?; // 数据是22050，这里也要存为22050

        Ok(())
    }
}
