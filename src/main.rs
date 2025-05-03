#![recursion_limit = "256"]

mod backend;
mod data;
mod inference;
mod model;
mod training;

use crate::training::TrainingConfig;
use backend::{AutodiffBackend, Backend};
use burn::{data::dataset::Dataset, optim::AdamConfig};
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
    let device = backend::device();
    let artifact_dir = "./artifacts";

    match args.mode {
        Mode::Train => {
            training::train::<AutodiffBackend>(
                artifact_dir,
                TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
                device.clone(),
            );
        }
        Mode::Infer => {
            inference::infer::<Backend>(
                artifact_dir,
                device,
                burn::data::dataset::vision::MnistDataset::test()
                    .get(42)
                    .unwrap(),
            );
        }
    }
}
