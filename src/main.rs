/// PARAS: 存储命令行参数的全局变量
use voice_clone::{
    error::MyError,
    parse_paras::parse_para,
    voice::convert_voice,
};

//#[tokio::main]
//async fn main() {

fn main() {
    if let Err(e) = run() {
        println!("{}", e); // 这里不要用`{:?}`，会打印结构体而不是打印指定的错误信息
    }
}

fn run() -> Result<(), MyError> {
    // 解析参数
    let paras = parse_para()?;

    // 语音转语音
    convert_voice(paras.voice, &paras.config, &paras.ckpt, paras.save)
}
