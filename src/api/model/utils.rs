use candle_core::{
    D,
    Device,
    DType,
    error::Error as candle_error,
    Shape,
    Tensor,
};
use candle_nn::{
    Conv1d,
    Conv1dConfig,
    Conv2d,
    Conv2dConfig,
    ConvTranspose1d,
    ConvTranspose1dConfig,
    Dropout,
    Module,
    ops::sigmoid,
    VarBuilder,
};

use crate::error::MyError;

pub fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    // P = ((S-1)*W-S+F)/2
    // For stride 1, this simplifies to (F-1)/2 for dilation 1.
    // With dilation: effective_kernel_size = (kernel_size - 1) * dilation + 1
    // padding = (effective_kernel_size - 1) / 2
    (kernel_size - 1) * dilation / 2
}

/// Enum to hold the specific configuration and kernel size for different convolution types
pub enum ConvOperationConfig {
    Conv1d {
        config: Conv1dConfig,
    },
    Conv2d {
        config: Conv2dConfig,
    },
    ConvTranspose1d {
        config: ConvTranspose1dConfig,
    },
}

/// 实现PyTorch的torch.nn.utils.weight_norm
pub struct ConvWeightNorm {
    // Parameters for weight normalization
    weight_v: Tensor, // Direction
    weight_g: Tensor, // Magnitude

    // Optional bias
    bias: Option<Tensor>,

    // 输入输出大小
    //out_channels: usize,

    // Specific convolution operation details
    op_config: ConvOperationConfig,

    // State for remove_weight_norm
    // When true, use precomputed_weight instead of g and v
    is_norm_removed: bool,
    precomputed_weight: Option<Tensor>, // Stores the fused weight after remove_weight_norm
}

impl ConvWeightNorm {
    /// Creates a new `ConvWeightNorm` layer for 1D convolution.
    pub fn new_conv1d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        Ok(Self {
            weight_v: vb.get((out_channels, in_channels / groups, kernel_size), "weight_v")?,
            weight_g: vb.get((out_channels, 1, 1), "weight_g")?,
            bias: if vb.contains_tensor("bias") {
                Some(vb.get((out_channels,), "bias")?)
            } else {
                None
            },
            //out_channels,
            op_config: ConvOperationConfig::Conv1d{
                config: Conv1dConfig{
                    padding,
                    stride,
                    dilation,
                    groups,
                    cudnn_fwd_algo: None,
                },
            },
            is_norm_removed: false,
            precomputed_weight: None,
        })
    }

    /// Creates a new `ConvWeightNorm` layer for 2D convolution.
    pub fn new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize), // (H, W)
        padding: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        Ok(Self {
            weight_v: vb.get((out_channels, in_channels / groups, kernel_size.0, kernel_size.1), "weight_v")?,
            weight_g: vb.get((out_channels, 1, 1, 1), "weight_g")?,
            bias: if vb.contains_tensor("bias") {
                Some(vb.get((out_channels,), "bias")?)
            } else {
                None
            },
            //out_channels,
            op_config: ConvOperationConfig::Conv2d{
                config: Conv2dConfig{
                    padding,
                    stride,
                    dilation,
                    groups,
                    cudnn_fwd_algo: None,
                },
            },
            is_norm_removed: false,
            precomputed_weight: None,
        })
    }

    /// Creates a new `ConvWeightNorm` layer for 1D transposed convolution.
    pub fn new_conv_transpose1d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        Ok(Self{
            weight_v: vb.get((in_channels / groups, out_channels, kernel_size), "weight_v")?,
            weight_g: vb.get((in_channels / groups, 1, 1), "weight_g")?,
            bias: if vb.contains_tensor("bias") {
                Some(vb.get((out_channels,), "bias")?)
            } else {
                None
            },
            //out_channels,
            op_config: ConvOperationConfig::ConvTranspose1d{
                config: ConvTranspose1dConfig{
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                },
            },
            is_norm_removed: false,
            precomputed_weight: None,
        })
    }

    /// Computes the effective weight tensor from `weight_g` and `weight_v`.
    /// w = g * (v / ||v||)
    /// `weight_v` has shape `(out_channels, in_channels, kernel_height, kernel_width)`
    /// `weight_g` has shape `(out_channels,)`
    /// We need to ensure `||v||` is computed per output filter and is not zero.
    /// https://arxiv.org/pdf/1602.07868
    fn compute_weight_new(&self) -> Result<Tensor, MyError> {
        // 计算每个输出通道的 L2 norm
        // 对于形状为 [out_channels, in_channels, kernel_height, kernel_width] 的张量
        // 我们需要在除第0维(out_channels)之外的所有维度上计算norm

        // 1. 先计算元素的平方
        let squared = self.weight_v.sqr()?;

        // 2. 在除第0维之外的所有维度上求和
        let weight_shape = self.weight_v.shape();
        let rank = self.weight_v.rank();
        let actual_out_channels = weight_shape.dims()[0]; // 从张量获取实际的out_channels

        let sum_dims: Vec<usize> = (1..rank).collect(); // [1, 2, 3] for 4D tensor
        let sum_squared = squared.sum(sum_dims)?;

        // 3. 开平方得到 L2 norm，形状应该是 [out_channels]
        let norm_v = sum_squared.sqrt()?;

        // 为了进行 broadcast_div，我们需要将 norm_v 的形状从 [out_channels]
        // 扩展为 [out_channels, 1, 1, 1]（对于4D张量）
        let mut norm_shape_vec = vec![actual_out_channels];
        for _ in 1..rank {
            norm_shape_vec.push(1);
        }
        let norm_v_expanded = norm_v.reshape(Shape::from(norm_shape_vec))?;

        // Normalize v: v / (||v|| + epsilon)
        // Epsilon for numerical stability, preventing division by zero
        let epsilon = 1e-12;
        let norm_v_with_epsilon = norm_v_expanded.affine(1.0, epsilon)?;
        let normalized_v = self.weight_v.broadcast_div(&norm_v_with_epsilon)?;

        // 同样地，将 weight_g 从 [out_channels] 扩展为 [out_channels, 1, 1, 1]
        let mut g_shape_vec = vec![actual_out_channels];
        for _ in 1..rank {
            g_shape_vec.push(1);
        }
        let g_expanded = self.weight_g.reshape(Shape::from(g_shape_vec))?;

        // effective_weight = g * normalized_v
        Ok(g_expanded.broadcast_mul(&normalized_v)?)
    }

    /// Gets the current effective weight, computing it if necessary or using the precomputed one.
    fn get_current_weight(&self) -> Result<Tensor, MyError> {
        if self.is_norm_removed {
            // .as_ref().unwrap() is safe due to is_norm_removed flag logic
            // Clone because conv ops take ownership or a reference, and we might need it again.
            // If precomputed_weight is used multiple times, cloning is necessary unless
            // the conv ops can take a reference that lives as long as `self`.
            // Candle's conv ops typically take `&Tensor`.
            Ok(self.precomputed_weight.as_ref().unwrap().clone())
        } else {
            self.compute_weight_new()
        }
    }

    /// Performs the forward pass for the `ConvWeightNorm` layer
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor, MyError> {
        let weight = self.get_current_weight()?; // This might be cloned or newly computed

        // https://docs.rs/candle-nn/0.9.1/candle_nn/conv/index.html
        match &self.op_config {
            ConvOperationConfig::Conv1d{config, ..} => {
                let conv_1d = Conv1d::new(weight, self.bias.clone(), *config);
                Ok(conv_1d.forward(xs)?)
            },
            ConvOperationConfig::Conv2d{config, ..} => {
                let conv_2d = Conv2d::new(weight, self.bias.clone(), *config);
                Ok(conv_2d.forward(xs)?)
            },
            ConvOperationConfig::ConvTranspose1d{config, ..} => {
                let conv_transpose_1d = ConvTranspose1d::new(weight, self.bias.clone(), *config);
                Ok(conv_transpose_1d.forward(xs)?)
            },
        }
    }
}

pub struct WeightNorm {
    hidden_channels: usize,
    //kernel_size:     usize,
    //dilation_rate:   f32,
    n_layers:        usize,
    //gin_channels:    usize,
    //p_dropout:       f32,
    in_layers:       Vec<ConvWeightNorm>,
    res_skip_layers: Vec<ConvWeightNorm>,
    drop:            Dropout,
    cond_layer:      Option<ConvWeightNorm>,
}

impl WeightNorm {
    pub fn new(
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: f32,
        n_layers: usize,
        gin_channels: usize,
        p_dropout: f32,
        vb: VarBuilder,
    ) -> Result<Self, MyError> {
        let vb_in_layers = vb.pp("in_layers");
        let mut in_layers = Vec::new();

        let vb_res_skip_layers = vb.pp("res_skip_layers");
        let mut res_skip_layers = Vec::new();

        for i in 0..n_layers {
            let dilation = dilation_rate.powi(i as i32) as usize;
            let padding = (kernel_size * dilation - dilation) / 2;
            in_layers.push(ConvWeightNorm::new_conv1d(
                hidden_channels, // in_channels
                2 * hidden_channels, // out_channels
                kernel_size, // kernel_size,
                padding, // padding
                dilation, // dilation
                1, // stride,
                1, // groups,
                vb_in_layers.pp(&format!("{}", i)), // 共16层，enc_q.enc.in_layers.xxx
            )?);
            res_skip_layers.push(ConvWeightNorm::new_conv1d(
                hidden_channels, // in_channels
                if i < n_layers - 1 { // last one is not necessary
                    2 * hidden_channels
                } else {
                    hidden_channels
                }, // out_channels
                1, // kernel_size,
                0, // padding
                1, // dilation
                1, // stride,
                1, // groups,
                vb_res_skip_layers.pp(&format!("{}", i)), // 共16层，enc_q.enc.res_skip_layers.xxx
            )?);
        }

        Ok(WeightNorm {
            hidden_channels,
            //kernel_size,
            //dilation_rate,
            n_layers,
            //gin_channels,
            //p_dropout,
            in_layers,
            res_skip_layers,
            drop: Dropout::new(p_dropout),
            cond_layer: if gin_channels != 0 {
                Some(ConvWeightNorm::new_conv1d(
                    gin_channels, // in_channels
                    2 * hidden_channels * n_layers, // out_channels
                    1, // kernel_size,
                    0, // padding
                    1, // dilation
                    1, // stride,
                    1, // groups,
                    vb.pp("cond_layer"),
                )?)
            } else {
                None
            },
        })
    }

    pub fn forward(&self, x_initial: &Tensor, x_mask: Option<Tensor>, g: Option<Tensor>) -> Result<Tensor, MyError> {
        let mut x = x_initial.clone(); // x will be modified in the loop
        let mut output = x.zeros_like()?;
        let g_processed: Option<Tensor> = match (&g, &self.cond_layer) {
            (Some(g_val), Some(cond_layer)) => Some(cond_layer.forward(g_val)?),
            (Some(g_val), None) => Some(g_val.clone()), // g is provided but no cond_layer
            (None, _) => None, // g is not provided
        };
        for i in 0..self.n_layers {
            let x_in = self.in_layers[i].forward(&x)?;
            let g_l = match &g_processed {
                Some(g_tensor) => {
                    // g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                    // The slicing is on dimension 1 (channels).
                    let cond_offset = i * 2 * self.hidden_channels;
                    // Ensure g_tensor has enough channels.
                    // Shape of g_tensor is expected to be (Batch, TotalCondChannels, Time)
                    g_tensor.narrow(1, cond_offset, 2 * self.hidden_channels)?
                },
                None => x_in.zeros_like()?,
            };

            let mut acts = fused_add_tanh_sigmoid_multiply(&x_in, &g_l, (self.hidden_channels,))?;
            acts = self.drop.forward(&acts, false)?; // 第2个参数是否train

            let res_skip_acts = self.res_skip_layers[i].forward(&acts)?;
            if i < self.n_layers - 1 {
                let res_acts = res_skip_acts.narrow(1, 0, self.hidden_channels)?;
                x = ((x.add(&res_acts))?.broadcast_mul(x_mask.as_ref().unwrap()))?;
                let skip_acts = res_skip_acts.narrow(1, self.hidden_channels, self.hidden_channels)?;
                output = output.add(&skip_acts)?;
            } else {
                output = output.add(&res_skip_acts)?;
            }
            let _ = output.broadcast_mul(x_mask.as_ref().unwrap());
        }
        Ok(output)
    }
}

/// 融合操作：加法 + tanh + sigmoid + 逐元素乘法
///
/// 该函数执行以下操作：
/// 1. 将两个输入张量相加
/// 2. 沿通道维度分割结果张量
/// 3. 对第一部分应用 tanh 激活
/// 4. 对第二部分应用 sigmoid 激活
/// 5. 将两个激活结果逐元素相乘
///
/// # 参数
/// - `input_a`：第一个输入张量，形状为 [B, C, ...]
/// - `input_b`：第二个输入张量，形状需与 input_a 相同
/// - `n_channels`：元组，包含单个元素 (channels,)，表示分割点
///
/// # 返回值
/// 返回融合操作后的结果张量，形状为 [B, n_channels, ...]
///
/// # 错误
/// 可能返回以下错误：
/// - 张量加法失败
/// - 通道维度大小不匹配
/// - 张量切片操作失败
/// - 激活函数应用失败
///
/// # 示例
/// ```
/// let a = Tensor::randn(&[2, 16, 128], &Device::Cpu)?;
/// let b = Tensor::randn(&[2, 16, 128], &Device::Cpu)?;
/// let result = fused_add_tanh_sigmoid_multiply(&a, &b, (8,))?;
/// ```
pub fn fused_add_tanh_sigmoid_multiply(input_a: &Tensor, input_b: &Tensor, n_channels: (usize,)) -> Result<Tensor, MyError> {
    let n_channels_int = n_channels.0;

    // 1. 输入张量相加
    //let in_act = (input_a + input_b)?;
    let in_act = input_a.broadcast_add(input_b)?;

    // 获取通道维度大小
    let channels = in_act.dim(D::Minus2)?;

    // 验证分割点有效性
    if n_channels_int > channels {
        return Err(MyError::CandleError(candle_error::Msg(format!(
            "分割点 {} 超过通道维度大小 {}",
            n_channels_int, channels
        ))));
    }

    // 2. 沿通道维度分割张量
    // 第一部分：0 到 n_channels_int (tanh 输入)
    let t_act = in_act
        .narrow(D::Minus2, 0, n_channels_int)?
        .tanh()?;

    // 第二部分：n_channels_int 到结束 (sigmoid 输入)
    let s_act = sigmoid(&in_act.narrow(D::Minus2, n_channels_int, channels - n_channels_int)?)?;

    // 3. 逐元素相乘并返回结果
    //Ok((t_act * s_act)?)
    Ok(t_act.mul(&s_act)?)
}

/// Creates a sequence mask
/// Given a 1D tensor of sequence lengths, this function generates a 2D boolean mask
/// indicating valid positions (1) and padding positions (0) up to a specified
/// maximum length.
/// Returns a `Result` containing a 2D `Tensor` of `U8` (0 or 1).
/// Shape: `(batch_size, max_length)`
pub fn sequence_mask(length: u32, max_length: Option<u32>, device: &Device) -> Result<Tensor, MyError> {
    let max_length = match max_length {
        Some(m) => m, // 2078
        None => length, // 2078
    };
    let x = Tensor::arange(0, max_length, device)?; // Creates a U32 tensor, 2078
    let x_expanded = x.unsqueeze(0)?; // Shape: (1, max_length)

    // Perform comparison: x_expanded < length
    // In Candle, comparison ops like `lt` return a U8 tensor (0 for false, 1 for true).
    Ok(x_expanded.lt(length)?.to_dtype(DType::F32)?) // Shape: (batch_size, max_len_val), pytorch代码这里返回的是bool, [1, 2078]
}
