use candle_core::{
    error::Error as candle_error,
    Tensor,
};
use candle_nn::{
    Conv1d,
    Conv1dConfig,
    Module,
    VarBuilder,
};

use crate::{
    api::model::utils::{
        sequence_mask,
        WeightNorm,
    },
    error::MyError,
};

pub struct PosteriorEncoder {
    //in_channels:     usize,
    out_channels:    usize,
    //hidden_channels: usize,
    //kernel_size:     usize,
    //dilation_rate:   f32,
    //n_layers:        usize,
    //gin_channels:    usize,
    pre:             Conv1d,
    enc:             WeightNorm,
    proj:            Conv1d,
}

impl PosteriorEncoder {
    pub fn new(
        //in_channels: usize,
        out_channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: f32,
        n_layers: usize,
        gin_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        let vb_pre = vb.pp("pre");
        let vb_proj = vb.pp("proj");
        Ok(PosteriorEncoder {
            //in_channels,
            out_channels,
            //hidden_channels,
            //kernel_size,
            //dilation_rate,
            //n_layers,
            //gin_channels,
            pre: Conv1d::new(
                vb_pre.get((192, 513, 1), "weight")?, // weight, enc_q.pre.weight
                Some(vb_pre.get(192, "bias")?), // bias, enc_q.pre.bias
                Conv1dConfig{ // config
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            enc: WeightNorm::new(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels,
                0.0, // p_dropout
                vb.pp("enc"),
            )?,
            proj: Conv1d::new(
                vb_proj.get((384, 192, 1), "weight")?, // weight, enc_q.proj.weight
                Some(vb_proj.get(384, "bias")?), // bias, enc_q.proj.bias
                Conv1dConfig{ // config
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
        })
    }

    pub fn forward(&self, x_initial: &Tensor, x_lengths: usize, g: Option<Tensor>, tau: f32) -> Result<(Tensor, Tensor, Tensor, Tensor), MyError> {
        //let x_mask = sequence_mask(x_lengths as u32, Some(x_initial.dims()[2] as u32), x_initial.device())?.unsqueeze(1)?;
        let x_mask = sequence_mask(x_lengths as u32, Some(x_initial.dims()[2] as u32), x_initial.device())?.unsqueeze(1)?; // [1, 2078], unsqueeze(1)之后: [1, 1, 2078]
        //println!("enc_q x_mask: {:?}\n{:?}", x_mask, x_mask.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?);

        let mut tmp_x_initial = self.pre.forward(x_initial)?; // [1, 192, 2078]
        //println!("tmp_x_initial 1: {:?}, {:?}", tmp_x_initial, tmp_x_initial.squeeze(0)?.to_vec2::<f32>()?);
        tmp_x_initial = tmp_x_initial.broadcast_mul(&x_mask)?; // [1, 192, 2078]
        //println!("tmp_x_initial 2: {:?}, {:?}", tmp_x_initial, tmp_x_initial.squeeze(0)?.to_vec2::<f32>()?);

        tmp_x_initial = self.enc.forward(&tmp_x_initial, Some(x_mask.clone()), g)?; // [1, 192, 2078]
        //println!("tmp_x_initial 3: {:?}, {:?}", tmp_x_initial, tmp_x_initial.squeeze(0)?.to_vec2::<f32>()?);

        let mut stats = self.proj.forward(&tmp_x_initial)?; // [1, 384, 2078]
        stats = stats.broadcast_mul(&x_mask)?; // [1, 384, 2078]
        //println!("stats: {:?}, {:?}", stats, stats.squeeze(0)?.to_vec2::<f32>()?);

        let chunks = stats.chunk(2, 1)?; // Split dimension 1 into 2 chunks
        if chunks.len() != 2 {
            return Err(MyError::CandleError(candle_error::Msg(format!(
                "Expected 2 chunks from stats.chunk, got {}",
                chunks.len()
            ))));
        }
        let m = &chunks[0]; // [1, 192, 2078]
        //println!("m: {:?}, {:?}", m, m.squeeze(0)?.to_vec2::<f32>()?);
        let logs = &chunks[1]; // [1, 192, 2078]
        //println!("logs: {:?}, {:?}", logs, logs.squeeze(0)?.to_vec2::<f32>()?);

        if m.dim(1)? != self.out_channels || logs.dim(1)? != self.out_channels {
            return Err(MyError::CandleError(candle_error::Msg(format!(
                "Chunk dimension mismatch: expected {}, got {} for m and {} for logs at dim 1",
                self.out_channels, m.dim(1)?, logs.dim(1)?
            ))));
        }

        // 下面计算`z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask`
        let device = x_initial.device();
        let noise = m.randn_like(0.0_f64, 1.0_f64)?.squeeze(0)?.to_vec2::<f32>()?.iter().map(|a| a.iter().map(|b| *b as f32).collect::<Vec<_>>()).collect::<Vec<_>>(); // [192, 2078]
        let noise = Tensor::new(noise, &device)?.unsqueeze(0)?; // [1, 192, 2078]
        //println!("noise: {:?}, {:?}", noise, noise.squeeze(0)?.to_vec2::<f32>()?);

        let exp_logs = logs.exp()?; // [1, 192, 2078]

        let scaled_exp_logs = exp_logs.broadcast_mul(&Tensor::new(tau, device)?)?; // [1, 192, 2078]
        //println!("scaled_exp_logs: {:?}, {:?}", scaled_exp_logs, scaled_exp_logs.squeeze(0)?.to_vec2::<f32>()?);

        let random_component = noise.broadcast_mul(&scaled_exp_logs)?; // [1, 192, 2078]

        let z_unmasked = m.broadcast_add(&random_component)?; // [1, 192, 2078]
        //let z_unmasked = m.broadcast_add(&scaled_exp_logs)?; // [1, 192, 2078]

        let z = z_unmasked.broadcast_mul(&x_mask)?; // [1, 192, 2078]

        Ok((z, m.clone(), logs.clone(), x_mask)) // Clone m and logs as they are borrowed from `chunks`
    }
}
