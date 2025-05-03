mod data;
mod inference;
mod model;
mod training;

use crate::training::TrainingConfig;
use burn::{
    backend::{Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::AdamConfig,
};
use clap::{Parser, ValueEnum};
use model::ModelConfig;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(value_enum)]
    mode: Mode,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    Train,
    Infer,
}

fn main() {
    let args = Args::parse();

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./artifacts";

    match args.mode {
        Mode::Train => {
            crate::training::train::<MyAutodiffBackend>(
                artifact_dir,
                TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
                device.clone(),
            );
        }
        Mode::Infer => {
            crate::inference::infer::<MyBackend>(
                artifact_dir,
                device,
                burn::data::dataset::vision::MnistDataset::test()
                    .get(42)
                    .unwrap(),
            );
        }
    }
}
