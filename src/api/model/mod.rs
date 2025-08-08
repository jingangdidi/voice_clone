// +--------------------------------------------------------------------------------------+
// | tone_color_converter --> open_voice_base --> synthesizer --> generator               | 音频解码器，最终生成目标音色音频，基于WaveNet结构的声码器，将潜在表示转换为原始音频波形
// |                                                              posterior_encoder       | 后验编码器，提取目标音频的后验分布，提取目标音频的后验分布参数（均值、方差）
// |                                                              residual_coupling_block | 流模型，实现流模型进行音色变换，使用可逆流模型进行音色变换，包含多个ResidualCouplingLayer
// |                                                              reference_encoder       | 提取参考音色特征，从频谱图提取音色特征，从频谱图中提取说话人特征，通过卷积+GRU结构实现
// +--------------------------------------------------------------------------------------+

pub mod utils;

pub mod generator;
pub mod posterior_encoder;
pub mod residual_coupling_block;
pub mod reference_encoder;

pub mod synthesizer;

pub mod open_voice_base;

pub mod tone_color_converter;
