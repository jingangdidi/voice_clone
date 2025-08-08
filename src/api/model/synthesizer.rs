use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::{
    api::model::{
        generator::Generator,
        posterior_encoder::PosteriorEncoder,
        residual_coupling_block::ResidualCouplingBlock,
        reference_encoder::ReferenceEncoder,
    },
    config::HParams,
    error::MyError,
};

/// Synthesizer for Training
pub struct SynthesizerTrn {
    dec:        Generator,
    enc_q:      PosteriorEncoder,
    flow:       ResidualCouplingBlock,
    pub ref_enc:    ReferenceEncoder,
    //n_speakers: usize,
    zero_g:     bool,
}

impl SynthesizerTrn {
    pub fn new(config: &HParams, vb: VarBuilder) -> Result<Self, MyError> {
        let spec_channels = config.data.filter_length / 2 + 1; // 1024 / 2 + 1 = 513
        Ok(SynthesizerTrn{
            dec: Generator::new(
                config.model.inter_channels, // initial_channel, 192
                config.model.resblock.parse::<usize>().unwrap(), // resblock, 1
                config.model.resblock_kernel_sizes, // resblock_kernel_sizes, [3, 7, 11]
                &config.model.resblock_dilation_sizes, // resblock_dilation_sizes, [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
                config.model.upsample_rates, // upsample_rates, [8, 8, 2, 2]
                config.model.upsample_initial_channel, // upsample_initial_channel, 512
                config.model.upsample_kernel_sizes, // upsample_kernel_sizes, [16, 16, 4, 4]
                config.model.gin_channels, // gin_channels, 256
                vb.pp("dec"),
            )?,
            enc_q: PosteriorEncoder::new(
                //spec_channels,
                config.model.inter_channels, // 192
                config.model.hidden_channels, // 192
                5,
                1.0,
                16,
                config.model.gin_channels, // 256
                vb.pp("enc_q"),
            )?,
            flow: ResidualCouplingBlock::new(
                config.model.inter_channels, // 192
                config.model.hidden_channels, // 192
                5, // kernel_size
                1, // dilation_rate
                4, // n_layers
                4, // nflows
                config.model.gin_channels, // 256
                vb.pp("flow"),
            )?,
            ref_enc: ReferenceEncoder::new(
                spec_channels,
                config.model.gin_channels,
                true, // false,
                vb.pp("ref_enc"),
            )?,
            //n_speakers: config.data.n_speakers,
            zero_g: config.model.zero_g,
        })
    }

    pub fn voice_conversion(
        &self,
        y: Tensor,
        y_lengths: usize,
        sid_src: &Tensor,
        sid_tgt: &Tensor,
        tau: f32,
    ) -> Result<(Tensor, Tensor, (Tensor, Tensor, Tensor)), MyError> {
        // z: [1, 192, 2078]
        // m_q: [1, 192, 2078]
        // logs_q: [1, 192, 2078]
        // y_mask: [1, 1, 2078]
        let (z, _m_q, _logs_q, y_mask) = self.enc_q.forward(
            &y,
            y_lengths,
            Some(if !self.zero_g {
                sid_src.clone()
            } else {
                sid_src.zeros_like()?
            }),
            tau,
        )?;
        //println!("z: {:?}, {:?}", z, z.squeeze(0)?.to_vec2::<f32>()?);
        //println!("m_q: {:?}, {:?}", m_q, m_q.squeeze(0)?.to_vec2::<f32>()?);
        //println!("logs_q: {:?}, {:?}", logs_q, logs_q.squeeze(0)?.to_vec2::<f32>()?);
        //println!("y_mask: {:?}, {:?}", y_mask, y_mask.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?);
        let z_p = self.flow.forward(&z, &y_mask, Some(sid_src.clone()), false)?; // [1, 192, 2078]
        //println!("z_p: {:?}, {:?}", z_p, z_p.squeeze(0)?.to_vec2::<f32>()?);
        let z_hat = self.flow.forward(&z_p, &y_mask, Some(sid_tgt.clone()), true)?; // [1, 192, 2078]
        //println!("z_hat: {:?}, {:?}", z_hat, z_hat.squeeze(0)?.to_vec2::<f32>()?);
        let o_hat = self.dec.forward( // [1, 1, 531968]
            &z_hat.broadcast_mul(&y_mask)?,
            Some(if !self.zero_g {
                sid_tgt.clone()
            } else {
                sid_tgt.zeros_like()?
            }),
        )?;
        //println!("o_hat: {:?}, {:?}", o_hat, &o_hat.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?[0..20]);
        Ok((o_hat, y_mask, (z, z_p, z_hat)))
    }
}
