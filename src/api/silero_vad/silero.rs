use ndarray::{Array, Array2, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

use crate::{
    api::silero_vad::utils,
    error::MyError,
};

static MODEL_BYTES: &[u8] = include_bytes!("../../../silero_vad/silero_vad.onnx");

#[derive(Debug)]
pub struct Silero {
    session: ort::Session,
    sample_rate: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    state: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    context: Option<[f32; 64]>, // 对应python版本代码，存储上次数据的最后64个数值
}

impl Silero {
    pub fn new(sample_rate: utils::SampleRate) -> Result<Self, MyError> {
        //let session = ort::Session::builder()?.commit_from_file(model_path)?;
        let session = ort::Session::builder()?.commit_from_memory(MODEL_BYTES)?;
        let state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        let sample_rate = Array::from_shape_vec([1], vec![sample_rate.into()]).unwrap();
        Ok(Self {
            session,
            sample_rate,
            state,
            context: None,
        })
    }

    pub fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
    }

    // rust这里相比python版简化了，python每次都把上一次的最后64个数据加到本次数据前面
    // 而rust没有做这一步，并且rust这里只获取数据的前480个值
    pub fn calc_level(&mut self, audio_frame: &[f32]) -> Result<f32, MyError> {
        // srx添加---开始
        let data: Vec<f32> = match self.context {
            Some(c) => c.into_iter().chain(audio_frame.to_vec().into_iter()).collect(), // 在数据前加上上次数据的最后64个值
            None => [0.0; 64].into_iter().chain(audio_frame.to_vec().into_iter()).collect(), // 在数据前加上64个0
        };
        self.context = { // 更新为本次数据的最后64个值
            //data.last_chunk::<64>().into()
            let mut context = [0.0; 64];
            context.copy_from_slice(&data[(data.len() - 64)..]);
            Some(context)
        };
        // srx添加---结束
        //println!("data_64_start: {:?}", &data[..64]);
        //println!("data_64_end: {:?}", &data[(data.len()-64)..]);
        let frame = Array2::<f32>::from_shape_vec([1, data.len()], data).unwrap();
        let inps = ort::inputs![
            frame,
            std::mem::take(&mut self.state),
            self.sample_rate.clone(),
        ]?;
        let res = self
            .session
            .run(ort::SessionInputs::ValueSlice::<3>(&inps))?;
        self.state = res["stateN"].try_extract_tensor().unwrap().to_owned();
        Ok(*res["output"]
            .try_extract_raw_tensor::<f32>()
            .unwrap()
            .1
            .first()
            .unwrap())
    }
}
