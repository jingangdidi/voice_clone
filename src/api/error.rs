use std::io;
use std::num::ParseFloatError;
use std::string::FromUtf8Error;

use candle_core::error::Error as candle_error;
use hound::Error as hound_error;

#[cfg(feature = "silero_vad")]
use ort::Error as ort_error;

use safetensors::tensor::SafeTensorError as safe_tensor_error;
use serde_json::Error as json_error;
use symphonia::core::errors::Error as symphonia_error;
use thiserror::Error;

/// srx添加，自定义的错误类型，方便传递错误
/// 参考：https://github.com/dtolnay/thiserror
/// 参考：https://crates.io/crates/thiserror
/// 参考：https://juejin.cn/post/7272005801081126968
/// 参考：https://www.shakacode.com/blog/thiserror-anyhow-or-how-i-handle-errors-in-rust-apps/
/// 参考：https://rustcc.cn/article?id=1e20f814-c7d5-4aca-bb67-45dcfb65d9f9
#[derive(Debug, Error)]
pub enum MyError {
    // 读取文件错误
    #[error("Error - fs::read {file}: {error}")]
    ReadFileError{file: String, error: io::Error},

    // 打开文件错误
    #[error("Error - fs::File::open {file}: {error}")]
    OpenFileError{file: String, error: io::Error},

    // 创建文件错误
    #[error("Error - fs::create {file}: {error}")]
    CreateFileError{file: String, error: io::Error},

    // 创建路径错误
    #[error("Error - fs::create_dir_all {dir_name}: {error}")]
    CreateDirAllError{dir_name: String, error: io::Error},

    // 创建文件(一次写入)错误
    #[error("Error - fs::write {file}: {error}")]
    WriteFileError{file: String, error: io::Error},

    // 按行读取文件错误
    #[error("Error - read lines {file}: {error}")]
    LinesError{file: String, error: io::Error},

    // 获取指定路径下所有项错误
    #[error("Error - read_dir {dir}: {error}")]
    ReadDirError{dir: String, error: io::Error},

    // 删除文件夹错误
    #[error("Error - fs::remove_dir {dir}: {error}")]
    RemoveDirError{dir: String, error: io::Error},

    // 删除文件错误
    #[error("Error - fs::remove_file {file}: {error}")]
    RemoveFileError{file: String, error: io::Error},

    // 读取文件到字符串错误
    #[error("Error - read_to_string {file}: {error}")]
    ReadToStringError{file: String, error: io::Error},

    // 字符串转指定类型错误
    #[error("Error - parse {from} -> {to}: {error}")]
    ParseStringError{from: String, to: String, error: ParseFloatError},

    // 路径不存在
    #[error("Error - {dir} does not exist")]
    DirNotExistError{dir: String},

    // 文件不存在
    #[error("Error - {file} does not exist")]
    FileNotExistError{file: String},

    // 读取文件转为UTF-8错误
    #[error("Error - {file} to UTF-8: {error}")]
    FileContentToUtf8Error{file: String, error: FromUtf8Error},

    // 数据转为json字符串错误
    #[error("Error - to json string: {error}")]
    ToJsonStirngError{uuid: String, error: json_error},

    // json转字符串错误
    #[error("Error - serde_json::to_string: {error}")]
    JsonToStringError{error: io::Error},

    // 字符串转json错误
    #[error("Error - string to json: {error}")]
    StringToJsonError{error: json_error},

    // candle错误
    #[error("Error - candle: {0}")]
    CandleError(#[from] candle_error),

    // 解码音频文件错误
    #[error("Error - load audio: {0}")]
    LoadAudioError(#[from] symphonia_error),

    // hound错误
    #[error("Error - hound: {0}")]
    HoundError(#[from] hound_error),

    // ort错误
    #[cfg(feature = "silero_vad")]
    #[error("Error - ort onnx: {0}")]
    OrtError(#[from] ort_error),

    // SafeTensor错误
    #[error("Error - SafeTensor error: {0}")]
    SafeTensorError(#[from] safe_tensor_error),

    // 参数使用错误
    #[error("Error - {para}")]
    ParaError{para: String},

    // 其他错误
    #[error("Error - {info}")]
    NormalError{info: String},

    // 常规io::Error，这里可以改为向上面那样将错误传过来，但不知道还能否使用`#[from]`
    #[error("I/O error occurred")]
    IoError(#[from] io::Error),
}
