use candle_core::{
    Tensor,
};
use candle_nn::{
    GRU,
    GRUConfig,
    LayerNorm,
    linear,
    Linear,
    Module,
    RNN,
    VarBuilder,
};

use crate::{
    api::model::utils::ConvWeightNorm,
    error::MyError,
};

pub struct ReferenceEncoder {
    spec_channels: usize,
    convs:         Vec<ConvWeightNorm>,
    gru:           GRU,
    proj:          Linear,
    layernorm:     Option<LayerNorm>,
}

impl ReferenceEncoder {
    pub fn new(spec_channels: usize, gin_channels: usize, layernorm: bool, vb: VarBuilder) -> Result<Self, MyError> {
        let ref_enc_filters = [32, 32, 64, 64, 128, 128];
        let filters = [1, 32, 32, 64, 64, 128, 128];

        let vb_convs = vb.pp("convs");
        let mut convs = Vec::new();
        for i in 0..6 {
            convs.push(ConvWeightNorm::new_conv2d(
                filters[i], // in_channels
                filters[i + 1], // out_channels
                (3, 3), // kernel_size, (w, h)
                1, // padding
                1, // dilation,
                2, // stride
                1, // groups
                vb_convs.pp(&format!("{}", i)), // ref_enc.convs.0, ref_enc.convs.1, ref_enc.convs.2, ref_enc.convs.3, ref_enc.convs.4, ref_enc.convs.5
            )?);
        }

        let out_channels = calculate_channels(spec_channels, 3, 2, 1, 6);
        let gru = GRU::new(
            ref_enc_filters[ref_enc_filters.len() - 1] * out_channels, // in_dim
            128, // hidden_dim
            GRUConfig::default(), // config
            vb.pp("gru"),
        )?;

        let proj = linear(
            128, // in_dim
            gin_channels, // out_dim
            vb.pp("proj"),
        )?;

        
        let layer_norm = if layernorm {
            let vb_layernorm = vb.pp("layernorm");
            Some(LayerNorm::new(
                vb_layernorm.get((513,), "weight")?,
                vb_layernorm.get((513,), "bias")?,
                1e-5,
            ))
        } else {
            None
        };

        Ok(ReferenceEncoder {
            spec_channels,
            convs,
            gru,
            proj,
            layernorm: layer_norm,
        })
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor, MyError> {
        let n = inputs.dim(0)?;
        //println!("ref_enc0: {:?}", inputs.squeeze(0)?.to_vec2::<f32>());

        // Reshape inputs: [n, 1, Ty, n_freqs]
        let mut out = inputs.reshape((n, 1, (), self.spec_channels))?;

        // Apply LayerNorm if present
        if let Some(ln) = &self.layernorm {
            out = ln.forward(&out)?;
            //println!("ref_enc: {:?}", out.squeeze(0)?.squeeze(0)?.to_vec2::<f32>());
        }

        // Apply convolution layers with ReLU
        for conv in &self.convs {
            out = conv.forward(&out)?;
            out = out.relu()?;
        } // 循环后得到[1, 128, 16, 9]
        //println!("ref_enc: {:?}, {:?}", out.dims(), out.squeeze(0)?.to_vec3::<f32>());

        // Transpose and reshape: [n, Ty//2^K, 128*n_mels//2^K]
        let out = out.transpose(1, 2)?; // [1, n, 128, 9]
        let t = out.dim(1)?;
        let n = out.dim(0)?;
        let out = out.reshape((n, t, ()))?; // [1, 16, 1152]
        //println!("ref_enc: {:?}, {:?}", out.dims(), out.squeeze(0)?.to_vec2::<f32>());

        // Apply GRU
        // https://docs.rs/candle-nn/0.9.1/candle_nn/rnn/struct.GRU.html#impl-RNN-for-GRU
        let gru_out = self.gru.seq(&out)?; // 不能使用forward方法，返回`Result<Vec<Self::State>>`
        // println!("{:?}", gru_out);
        /*
        let out: Vec<Tensor> = gru_out.into_iter().map(|state| state.h).collect();
        // 在序列维度上连接: [n, 128]
        let out = Tensor::cat(&out, 0)?;
        */
        // `pytorch.gru`返回2个结果`output`和`h_n`，其中`h_n`就是最后一个hidden state
        // `gru.seq`返回了n个`GRUState { h: Tensor[dims 1, 128; f32] }`，这里仅获取最后一个
        let out = gru_out.last().unwrap().h.clone(); // [1, 128]
        //println!("ref_enc: {:?}, {:?}", out.dims(), out.squeeze(0)?.to_vec1::<f32>());

        // Project output
        Ok(self.proj.forward(&out)?)
    }
}

fn calculate_channels(l_initial: usize, kernel_size: usize, stride: usize, pad: usize, n_convs: usize) -> usize {
    let mut l = l_initial;
    for _ in 0..n_convs {
        l = (l - kernel_size + 2 * pad) / stride + 1
    }
    l
}
