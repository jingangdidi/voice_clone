use candle_core::{
    DType,
    Tensor,
};
use candle_nn::{
    Conv1d,
    Conv1dConfig,
    Module,
    VarBuilder,
};

use crate::{
    api::model::utils::WeightNorm,
    error::MyError,
};

struct ResidualCouplingLayer {
    //channels:        usize,
    //hidden_channels: usize,
    //kernel_size:     usize,
    //dilation_rate:   usize,
    //n_layers:        usize,
    half_channels:   usize,
    mean_only:       bool,
    pre:             Conv1d,
    enc:             WeightNorm,
    post:            Conv1d,
}

impl ResidualCouplingLayer {
    fn new(
        channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: usize,
        n_layers: usize,
        p_dropout: f32,
        gin_channels: usize,
        mean_only: bool,
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        let vb_pre = vb.pp("pre");
        let vb_post = vb.pp("post");

        Ok(ResidualCouplingLayer {
            //channels,
            //hidden_channels,
            //kernel_size,
            //dilation_rate,
            //n_layers,
            half_channels: channels / 2,
            mean_only,
            pre: Conv1d::new(
                vb_pre.get((192, 96, 1), "weight")?, // weight, flow.flows.xxx.pre.weight
                Some(vb_pre.get(192, "bias")?), // bias, flow.flows.xxx.pre.bias
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
                dilation_rate as f32,
                n_layers,
                gin_channels,
                p_dropout,
                vb.pp("enc"),
            )?,
            post: Conv1d::new(
                vb_post.get((96, 192, 1), "weight")?, // weight, flow.flows.xxx.post.weight
                Some(vb_post.get(96, "bias")?), // bias, flow.flows.xxx.post.bias
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

    fn forward(&self, x: &Tensor, x_mask: &Tensor, g: Option<Tensor>, reverse: bool) -> Result<(Tensor, Option<Tensor>), MyError> {
        // 第一部分：从第1维channels取前半部分
        let x0 = x.narrow(1, 0, self.half_channels)?;

        // 第二部分：从第1维channels取后半部分
        let x1 = x.narrow(1, self.half_channels, self.half_channels)?;

        // Pre-processing
        let h = self.pre.forward(&x0)?;
        let h = h.broadcast_mul(x_mask)?;

        // Encoder
        let h = self.enc.forward(&h, Some(x_mask.clone()), g)?;

        // Post-processing
        let stats = self.post.forward(&h)?;
        let stats = stats.broadcast_mul(x_mask)?;

        // Split stats into mean and log-scale
        let (m, logs) = if !self.mean_only {
            // 第一部分：从第1维channels取前半部分
            let splits_1 = stats.narrow(1, 0, self.half_channels)?;
            // 第二部分：从第1维channels取后半部分
            let splits_2 = stats.narrow(1, self.half_channels, self.half_channels)?;
            (splits_1, splits_2)
        } else {
            let zeros = Tensor::zeros_like(&stats)?;
            (stats, zeros)
        };

        if !reverse {
            // Forward transformation
            let exp_logs = logs.exp()?;
            let scaled_x1 = x1.mul(&exp_logs)?;
            let scaled_x1 = scaled_x1.broadcast_mul(x_mask)?;
            let transformed_x1 = m.add(&scaled_x1)?;

            // Concatenate back along channel dimension
            let x = Tensor::cat(&[x0, transformed_x1], 1)?;

            // Calculate log determinant
            let logdet = logs.sum_keepdim([1, 2])?;

            Ok((x, Some(logdet)))
        } else {
            // Inverse transformation
            let neg_logs = logs.neg()?;
            let exp_neg_logs = neg_logs.exp()?;
            let diff = x1.sub(&m)?;
            let scaled_diff = diff.mul(&exp_neg_logs)?;
            let transformed_x1 = scaled_diff.broadcast_mul(x_mask)?;

            // Concatenate back along channel dimension
            let x = Tensor::cat(&[x0, transformed_x1], 1)?;

            Ok((x, None))
        }
    }
}

struct Flip;

impl Flip {
    fn forward(&self, x: &Tensor, reverse: bool) -> Result<(Tensor, Option<Tensor>), MyError> {
        let x = x.flip(&[1])?;
        if !reverse {
            let logdet = Tensor::zeros(x.dims()[0], DType::F32, &x.device())?;
            Ok((x, Some(logdet)))
        } else {
            Ok((x, None))
        }
    }
}

pub struct ResidualCouplingBlock {
    //channels:        usize,
    //hidden_channels: usize,
    //kernel_size:     usize,
    //dilation_rate:   usize,
    //n_layers:        usize,
    //n_flows:         usize,
    //gin_channels:    usize,
    flows:           Vec<(ResidualCouplingLayer, Flip)>,
}

impl ResidualCouplingBlock {
    pub fn new(
        channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: usize,
        n_layers: usize,
        n_flows: usize,
        gin_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        let vb_flows = vb.pp("flows");
        let mut flows = Vec::new();
        for i in 0..n_flows {
            let tmp_rc_layer = ResidualCouplingLayer::new(
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                0.0, // p_dropout
                gin_channels,
                true, // mean_only
                vb_flows.pp(&format!("{}", 2*i)), // flow.flows.0, flow.flows.2, flow.flows.4, flow.flows.6
            )?;
            flows.push((tmp_rc_layer, Flip));
        }
        Ok(ResidualCouplingBlock {
            //channels,
            //hidden_channels,
            //kernel_size,
            //dilation_rate,
            //n_layers,
            //n_flows,
            //gin_channels,
            flows,
        })
    }

    pub fn forward(&self, x_initial: &Tensor, x_mask: &Tensor, g: Option<Tensor>, reverse: bool) -> Result<Tensor, MyError> {
        let mut x = x_initial.clone(); // Clone to allow mutable-like operations (re-assignment)
        if !reverse {
            for flow in &self.flows {
                x = flow.0.forward(&x, x_mask, g.clone(), reverse)?.0;
                x = flow.1.forward(&x, reverse)?.0;
            }
        } else {
            for flow in self.flows.iter().rev() {
                x = flow.1.forward(&x, reverse)?.0;
                x = flow.0.forward(&x, x_mask, g.clone(), reverse)?.0;
            }
        }
        Ok(x)
    }
}
