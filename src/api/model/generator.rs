use candle_core::{
    Tensor,
};
use candle_nn::{
    Conv1d,
    Conv1dConfig,
    Module,
    ops::leaky_relu,
    VarBuilder,
};
use rayon::prelude::*;

use crate::{
    api::model::utils::{
        ConvWeightNorm,
        get_padding,
    },
    error::MyError,
};

const LRELU_SLOPE: f64 = 0.1;

pub struct ResBlock1 {
    convs1: Vec<ConvWeightNorm>,
    convs2: Vec<ConvWeightNorm>,
}

impl ResBlock1 {
    /// 创建ResBlock1
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilation: [usize; 3],
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        // convs1
        let vb_convs1 = vb.pp("convs1");
        let mut convs1 = Vec::new();
        for (i, &d) in dilation.iter().enumerate() {
            convs1.push(ConvWeightNorm::new_conv1d(
                channels, // in_channels
                channels, // out_channels
                kernel_size,
                get_padding(kernel_size, d), // padding
                d, // dilation
                1, // stride
                1, // groups
                vb_convs1.pp(&format!("{}", i)), // dec.resblocks.0.convs1.0, dec.resblocks.0.convs1.1, dec.resblocks.0.convs1.2
            )?);
        }

        // convs2, all convs2 have dilation 1
        let vb_convs2 = vb.pp("convs2");
        let mut convs2 = Vec::new();
        for i in 0..3 {
            convs2.push(ConvWeightNorm::new_conv1d(
                channels, // in_channels
                channels, // out_channels
                kernel_size,
                get_padding(kernel_size, 1), // padding
                1, // dilation
                1, // stride
                1, // groups
                vb_convs2.pp(&format!("{}", i)), // dec.resblocks.0.convs2.0, dec.resblocks.0.convs2.1, dec.resblocks.0.convs2.2
            )?);
        }

        Ok(Self{convs1, convs2})
    }

    /// Performs the forward pass for `ResBlock1`
    pub fn forward(&self, x_initial: &Tensor, x_mask: Option<Tensor>) -> Result<Tensor, MyError> {
        let mut x = x_initial.clone(); // Clone to allow mutable-like operations (re-assignment)
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let mut xt = leaky_relu(&x, LRELU_SLOPE)?;
            if let Some(ref mask) = x_mask {
                xt = xt.broadcast_mul(mask)?;
            }
            xt = c1.forward(&xt)?;
            xt = leaky_relu(&xt, LRELU_SLOPE)?;
            if let Some(ref mask) = x_mask {
                xt = xt.broadcast_mul(mask)?;
            }
            xt = c2.forward(&xt)?;
            x = (xt + x)?; // Residual connection
        }
        if let Some(mask) = x_mask {
            x = x.broadcast_mul(&mask)?;
        }
        Ok(x)
    }
}

pub struct Generator {
    num_kernels:   usize,
    num_upsamples: usize,
    conv_pre:      Conv1d,
    ups:           Vec<ConvWeightNorm>,
    resblocks:     Vec<ResBlock1>,
    conv_post:     Conv1d,
    cond:          Option<Conv1d>,
}

impl Generator {
    pub fn new(
        _initial_channel: usize, // 192
        _resblock: usize, // 1
        resblock_kernel_sizes: [usize; 3], // [3, 7, 11]
        resblock_dilation_sizes: &[[usize; 3]], // [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        upsample_rates: [usize; 4], // [8, 8, 2, 2]
        upsample_initial_channel: usize, // 512
        upsample_kernel_sizes: [usize; 4], // [16, 16, 4, 4]
        gin_channels: usize, // 256
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        // conv_pre
        // https://docs.rs/candle-nn/0.9.1/candle_nn/conv/struct.Conv1d.html
        // https://docs.rs/candle-nn/0.9.1/candle_nn/conv/struct.Conv1dConfig.html
        let vb_conv_pre = vb.pp("conv_pre");
        let conv_pre = Conv1d::new(
            vb_conv_pre.get((512, 192, 7), "weight")?, // weight, dec.conv_pre.weight
            Some(vb_conv_pre.get(512, "bias")?), // bias, dec.conv_pre.bias
            Conv1dConfig{ // config
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );

        // ups
        let vb_ups = vb.pp("ups");
        let mut ups = Vec::new();
        for (i, (u, k)) in upsample_rates.iter().zip(upsample_kernel_sizes.iter()).enumerate() {
            ups.push(ConvWeightNorm::new_conv_transpose1d(
                upsample_initial_channel / 2usize.pow(i as u32), // in_channels
                upsample_initial_channel / 2usize.pow(i as u32 + 1), // out_channels
                *k, // kernel_size
                (k - u) / 2, // padding
                0, // output_padding
                1, // dilation
                *u, // stride
                1, // groups
                vb_ups.pp(&format!("{}", i)), // dec.ups.0, dec.ups.1, dec.ups.2
            )?);
        }

        // resblocks
        let vb_resblock = vb.pp("resblocks");
        let mut idx = 0;
        let mut resblocks = Vec::new();
        for i in 0..ups.len() {
            let ch = upsample_initial_channel / 2usize.pow(i as u32 + 1);
            for (k, d) in resblock_kernel_sizes.iter().zip(resblock_dilation_sizes.iter()) {
                resblocks.push(ResBlock1::new(
                    ch, // channels
                    *k, // kernel_size
                    *d, // dilation
                    vb_resblock.pp(&format!("{}", idx)), // vb: VarBuilder
                )?);
                idx += 1;
            }
        }

        // conv_post
        let conv_post = Conv1d::new(
            vb.pp("conv_post").get((1, 32, 7), "weight")?, // weight, dec.conv_post.weight
            None, // bias
            Conv1dConfig{ // config
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );

        // cond
        let cond = if gin_channels != 0 {
            let vb_cond = vb.pp("cond");
            Some(Conv1d::new(
                vb_cond.get((512, 256, 1), "weight")?, // weight, dec.cond.weight
                Some(vb_cond.get(512, "bias")?), // bias, dec.cond.bias
                Conv1dConfig{ // config
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ))
        } else {
            None
        };

        Ok(Generator{
            num_kernels: resblock_kernel_sizes.len(),
            num_upsamples: upsample_rates.len(),
            conv_pre,
            ups,
            resblocks,
            conv_post,
            cond,
        })
    }

    pub fn forward(&self, x_initial: &Tensor, g: Option<Tensor>) -> Result<Tensor, MyError> {
        let mut x = self.conv_pre.forward(x_initial)?;
        //println!("Generator: {:?}, {:?}", x, x.squeeze(0)?.to_vec2::<f32>()?);
        if let Some(g) = g {
            if let Some(cond) = &self.cond {
                let cond_out = cond.forward(&g)?;
                x = x.broadcast_add(&cond_out)?;
                //println!("Generator: {:?}, {:?}", x, x.squeeze(0)?.to_vec2::<f32>()?);
            }
        }

        for i in 0..self.num_upsamples {
            // Apply leaky ReLU
            x = leaky_relu(&x, LRELU_SLOPE)?;
            //println!("{i}, Generator: {:?}, {:?}", x, x.squeeze(0)?.to_vec2::<f32>()?);

            // Upsample
            x = self.ups[i].forward(&x)?;

            // Process through residual blocks
            // 由于这里的循环最耗时，且循环没有先后依赖，因此可以使用rayon并行计算
            /*
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let res_output = self.resblocks[i * self.num_kernels + j].forward(&x, None)?;
                xs = match xs {
                    None => Some(res_output),
                    Some(prev) => Some((&prev + &res_output)?),
                };
            }
            */

            // 这里使用rayon并行计算
            let all_res_output: Vec<Tensor> = (0..self.num_kernels).into_par_iter().map(|j| self.resblocks[i * self.num_kernels + j].forward(&x, None).unwrap()).collect();
            let mut xs: Tensor = all_res_output[0].clone();
            for j in &all_res_output[1..] {
                xs = (xs + j)?;
            }
            let xs = Some(xs);

            // Average the residual outputs
            x = xs.unwrap().affine(1.0 / self.num_kernels as f64, 0.0)?;
        }
        //println!("Generator: {:?}, {:?}", x, &x.squeeze(0)?.to_vec2::<f32>()?[0][0..20]);

        // Final processing
        x = leaky_relu(&x, 0.01)?; // 这里的LRELU_SLOPE是0.01
        //println!("Generator: {:?}, {:?}", x, &x.squeeze(0)?.to_vec2::<f32>()?[0][0..20]);
        x = self.conv_post.forward(&x)?;
        //println!("Generator: {:?}, {:?}", x, &x.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?[0..20]);
        x = x.tanh()?;
        //println!("Generator: {:?}, {:?}", x, &x.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?[0..20]);

        Ok(x)
    }
}
