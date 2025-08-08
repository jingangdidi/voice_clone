use std::env::set_var;
use std::fs::create_dir_all;
use std::path::PathBuf;

use argh::FromArgs;
use candle_core::utils::cuda_is_available;

use crate::{
    api::base_speaker::TargetType,
    error::MyError,
};

#[derive(FromArgs)]
#[argh(help_triggers("-h", "--help"))] // https://github.com/google/argh/pull/106
/// voice clone
struct Paras {
    /// source files, colon separated
    #[argh(option, short = 's')]
    source: String,

    /// target files, colon separated. -t also support base speakers: en-au, en-br, en-default, en-india, en-newest, en-us, es, fr, jp, kr, zh. default: en-default
    #[argh(option, short = 't')]
    target: Option<String>,

    /// result voice file names, colon separated, default: source--target.wav
    #[argh(option, short = 'n')]
    name: Option<String>,

    /// openvoice model path, default: ./checkpoints_v2/converter
    #[argh(option, short = 'm')]
    model: Option<String>,

    /// save source and target tone color to to the same directory as the specified -s and -t files, maintaining identical nomenclature while altering the format extension to ".tone"
    #[argh(switch, short = 'S')]
    save: bool,

    /// cpu threads, 0 means all threads, default: 4
    #[argh(option, short = 'T')]
    thread: Option<usize>,

    /// output path, default: ./
    #[argh(option, short = 'o')]
    outpath: Option<String>,
}

/// 输入输出语音文件
pub struct Voice {
    pub source:   PathBuf,         // 原始语音文件，支持mp3、mp4、wav等格式，多个之间`:`间隔
    pub target:   TargetType,      // 内置的音色，或自己准备的语音文件，支持mp3、mp4、wav等格式，多个之间`:`间隔，不指定该参数则使用内置en-default
    pub src_name: String,          // 原始语音文件名，不含路径前缀和格式后缀
    pub tgt_name: String,          // 自己准备的语音文件名，不含路径前缀和格式后缀
    pub src_path: String,          // 原始语音文件名，含路径前缀和格式后缀，会作为HashMap的key
    pub tgt_path: String,          // 自己准备的语音文件名，含路径前缀和格式后缀，或内置的base speaker名称，会作为HashMap的key
    pub tone_s:   PathBuf,         // 原始语音对应的音色文件（与source同路径同名，只是格式后缀改为tone），如果该文件存在则直接读取，否则从source中提取，如果指定了-c，则将提取的音色保存至该文件
    pub tone_t:   Option<PathBuf>, // 自己准备的语音对应的音色文件（与target同路径同名，只是格式后缀改为tone），如果该文件存在则直接读取，否则从target中提取，如果指定了-c，则将提取的音色保存至该文件。如果使用内置的base音色则为None
    pub out_file: PathBuf,         // 最终转换生成的语音文件（含有路径，默认名称`source名称--target名称.wav`）
}

/// 存储解析后的命令行参数
///#[derive(Debug, Default)]
pub struct ParsedParas {
    pub voice:  Vec<Voice>, // source、target、最终输出的语音文件
    pub config: PathBuf,    // 模型config.json，默认`checkpoints_v2/converter/config.json`
    pub ckpt:   PathBuf,    // 模型checkpoint.pth，默认`checkpoints_v2/converter/checkpoint.pth`
    pub save:   bool,       // 保存source和target的音色至同路径下，且文件名相同，只是格式后缀为`pth`，方便下次使用时直接读取已经够提取的音色
}

/// 解析参数
pub fn parse_para() -> Result<ParsedParas, MyError> {
    let para: Paras = argh::from_env();
    // 先检查输出路径，不存在则创建
    let outpath = match para.outpath {
        Some(o) => check_outpath(&o)?,
        None => check_outpath("./")?,
    };
    // 检查source和target语音文件
    let mut source_files: Vec<&str> = para.source.split(":").collect();
    let mut target_files: Vec<TargetType> = match para.target {
        Some(t) => {
            match t.as_ref() {
                "en-au"      => vec![TargetType::EnAu],
                "en-br"      => vec![TargetType::EnBr],
                "en-default" => vec![TargetType::EnDefault],
                "en-india"   => vec![TargetType::EnIndia],
                "en-newest"  => vec![TargetType::EnNewest],
                "en-us"      => vec![TargetType::EnUs],
                "es"         => vec![TargetType::Es],
                "fr"         => vec![TargetType::Fr],
                "jp"         => vec![TargetType::Jp],
                "kr"         => vec![TargetType::Kr],
                "zh"         => vec![TargetType::Zh],
                _            => {
                    let mut target: Vec<TargetType> = Vec::new();
                    for i in t.split(":") {
                        match i {
                            "en-au"      => target.push(TargetType::EnAu),
                            "en-br"      => target.push(TargetType::EnBr),
                            "en-default" => target.push(TargetType::EnDefault),
                            "en-india"   => target.push(TargetType::EnIndia),
                            "en-newest"  => target.push(TargetType::EnNewest),
                            "en-us"      => target.push(TargetType::EnUs),
                            "es"         => target.push(TargetType::Es),
                            "fr"         => target.push(TargetType::Fr),
                            "jp"         => target.push(TargetType::Jp),
                            "kr"         => target.push(TargetType::Kr),
                            "zh"         => target.push(TargetType::Zh),
                            _            => {
                                let tmp_file = PathBuf::from(i);
                                if !(tmp_file.exists() && tmp_file.is_file()) {
                                    return Err(MyError::FileNotExistError{file: i.to_string()})
                                }
                                target.push(TargetType::File(tmp_file));
                            },
                        }
                    }
                    target
                },
            }
        },
        None => vec![TargetType::EnDefault],
    };
    let target_len = target_files.len();
    let (src_len, tgt_len) = (source_files.len(), target_len);
    if src_len != 1 && tgt_len != 1 && src_len != tgt_len {
        return Err(MyError::ParaError{para: format!("if -s != 1 and -t != 1, -s ({}) and -t ({}) must have same number", src_len, tgt_len)})
    }
    let mut voice_files: Vec<Voice> = vec![];
    (source_files, target_files) = match (src_len == 1, tgt_len == 1) {
        (true,  true)  => (source_files, target_files),
        (true,  false) => ((0..tgt_len).map(|_| source_files[0]).collect(), target_files),
        (false, true)  => (source_files, (0..src_len).map(|_| target_files[0].clone()).collect()),
        (false, false) => (source_files, target_files),
    };
    for (s, t) in source_files.iter().zip(target_files.into_iter()) {
        let (source, src_name, src_path, tone_s) = check_source_voice_file(s)?;
        let (target, tgt_name, tgt_path, tone_t) = check_target_voice_file(t)?;
        voice_files.push(Voice{
            source,
            target,
            src_name,
            tgt_name,
            src_path,
            tgt_path,
            tone_s,
            tone_t,
            out_file: outpath.clone(), // 先设为输出路径，后面解析名称再加上具体名称
        });
    }
    // 解析输出语音文件名，默认`source名称--target名称.wav`
    match para.name {
        Some(n) => {
            let names: Vec<&str> = n.split(":").collect();
            if names.len() != voice_files.len() {
                return Err(MyError::ParaError{para: format!("-n ({}) and max(-s ({}), -t ({})) must have same number", names.len(), source_files.len(), target_len)})
            }
            for (i, name) in names.iter().enumerate() {
                if !name.to_lowercase().ends_with(".wav") {
                    return Err(MyError::ParaError{para: "-n only support wav format voice file".to_string()})
                }
                voice_files[i].out_file.push(name); // 这里在输出路径基础上加上文件名
            }
        },
        None => for i in 0..voice_files.len() {
            let name = format!("{}--{}.wav", voice_files[i].src_name, voice_files[i].tgt_name);
            voice_files[i].out_file.push(name); // 这里在输出路径基础上加上文件名，默认`source名称--target名称.wav`
        },
    }
    // 获取openvoice模型文件
    let (config, ckpt) = check_openvoice_model(para.model)?;
    // 设置线程数，默认4，0表示使用当前可用的所有线程
    match para.thread { // 不设置环境变量时，Rayon使用的线程池默认为系统的逻辑核心数
        Some(t) => if t > 0 {
            if cuda_is_available() { // 使用cuda时-T无效
                println!("Warning: -T is invalid when using cuda");
            } else {
                set_var("RAYON_NUM_THREADS", t.to_string()); // 修改`RAYON_NUM_THREADS`环境变量，设置Rayon并行的线程数为-T指定线程数
            }
        },
        None => if !cuda_is_available() { // 使用cuda时-T无效
            set_var("RAYON_NUM_THREADS", "4"); // 默认4线程
        },
    }
    // 返回参数
    Ok(ParsedParas{
        voice: voice_files,
        config,
        ckpt,
        save: para.save, // 保存source和target的音色至同路径下，且文件名相同，只是格式后缀为`.tone`，方便下次使用时直接读取已经够提取的音色
    })
}

/// 检查输出路径，不存在则创建
fn check_outpath(path: &str) ->Result<PathBuf, MyError> {
    let tmp_path = PathBuf::from(path);
    if !(tmp_path.exists() && tmp_path.is_dir()) {
        if let Err(err) = create_dir_all(&tmp_path) {
            return Err(MyError::CreateDirAllError{dir_name: path.to_string(), error: err})
        }
    }
    Ok(tmp_path)
}

/// 检查指定的source语音文件格式后缀以及是否存在，返回(PathBuf, 去除路径前缀和格式后缀的名称字符串)
fn check_source_voice_file(file: &str) -> Result<(PathBuf, String, String, PathBuf), MyError> {
    /*
    if !file.to_lowercase().ends_with(".wav") {
        return Err(MyError::ParaError{para: "-s and -t only support wav format voice file".to_string()})
    }
    */
    let tmp_file = PathBuf::from(file);
    if !(tmp_file.exists() && tmp_file.is_file()) {
        return Err(MyError::FileNotExistError{file: file.to_string()})
    }
    let name = tmp_file.file_name().unwrap().to_str().unwrap().rsplitn(2, '.').collect::<Vec<&str>>()[1].to_string(); // test.wav --> [wav, test]
    let tmp_tone_file = tmp_file.with_extension("tone");
    Ok((tmp_file, name, file.to_string(), tmp_tone_file))
}

/// 检查指定的target语音文件格式后缀以及是否存在，返回(PathBuf, 去除路径前缀和格式后缀的名称字符串)
fn check_target_voice_file(target: TargetType) -> Result<(TargetType, String, String, Option<PathBuf>), MyError> {
    match target {
        TargetType::EnAu      => Ok((TargetType::EnAu,      "en-au".to_string(),      "en-au".to_string(),      None)),
        TargetType::EnBr      => Ok((TargetType::EnBr,      "en-br".to_string(),      "en-br".to_string(),      None)),
        TargetType::EnDefault => Ok((TargetType::EnDefault, "en-default".to_string(), "en-default".to_string(), None)),
        TargetType::EnIndia   => Ok((TargetType::EnIndia,   "en-india".to_string(),   "en-india".to_string(),   None)),
        TargetType::EnNewest  => Ok((TargetType::EnNewest,  "en-newest".to_string(),  "en-newest".to_string(),  None)),
        TargetType::EnUs      => Ok((TargetType::EnUs,      "en-us".to_string(),      "en-us".to_string(),      None)),
        TargetType::Es        => Ok((TargetType::Es,        "es".to_string(),         "es".to_string(),         None)),
        TargetType::Fr        => Ok((TargetType::Fr,        "fr".to_string(),         "fr".to_string(),         None)),
        TargetType::Jp        => Ok((TargetType::Jp,        "jp".to_string(),         "jp".to_string(),         None)),
        TargetType::Kr        => Ok((TargetType::Kr,        "kr".to_string(),         "kr".to_string(),         None)),
        TargetType::Zh        => Ok((TargetType::Zh,        "zh".to_string(),         "zh".to_string(),         None)),
        TargetType::File(t)   => {
            let file = t.to_str().unwrap().to_string();
            /*
            if !file.to_lowercase().ends_with(".wav") {
                return Err(MyError::ParaError{para: "-s and -t only support wav format voice file".to_string()})
            }
            */
            if !(t.exists() && t.is_file()) {
                return Err(MyError::FileNotExistError{file: file.clone()})
            }
            let name = t.file_name().unwrap().to_str().unwrap().rsplitn(2, '.').collect::<Vec<&str>>()[1].to_string(); // test.wav --> [wav, test]
            let tmp_tone_file = t.with_extension("tone");
            Ok((TargetType::File(t), name, file, Some(tmp_tone_file)))
        },
    }
}

/// 检查指定的模型路径下所需文件是否存在，返回config.json和checkpoint.pth的PathBuf
fn check_openvoice_model(path: Option<String>) -> Result<(PathBuf, PathBuf), MyError> {
    // 检查模型路径
    let tmp_path = {
        let (tmp_path, p_str) = match path {
            Some(p) => (PathBuf::from(&p), p),
            None => {
                let tmp_path = "./checkpoints_v2/converter/".to_string();
                (PathBuf::from(&tmp_path), tmp_path)
            },
        };
        if !(tmp_path.exists() && tmp_path.is_dir()) {
            return Err(MyError::DirNotExistError{dir: p_str})
        }
        tmp_path
    };
    // 检查config.json
    let mut tmp_config = tmp_path.clone();
    tmp_config.push("config.json");
    if !(tmp_config.exists() && tmp_config.is_file()) {
        return Err(MyError::FileNotExistError{file: tmp_config.to_str().unwrap().to_string()})
    }
    // 检查checkpoint.pth
    let mut tmp_checkpoint = tmp_path.clone();
    tmp_checkpoint.push("checkpoint.pth");
    if !(tmp_checkpoint.exists() && tmp_checkpoint.is_file()) {
        return Err(MyError::FileNotExistError{file: tmp_checkpoint.to_str().unwrap().to_string()})
    }
    Ok((tmp_config, tmp_checkpoint))
}
