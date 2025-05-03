#![allow(dead_code)]

// ------ CUDA backend ------ //
#[cfg(feature = "cuda")]
pub type Backend = burn::backend::Cuda<f32, i32>;
#[cfg(feature = "cuda")]
pub type AutodiffBackend = burn::backend::Autodiff<Backend>;
#[cfg(feature = "cuda")]
pub fn device() -> burn::backend::cuda::CudaDevice {
    burn::backend::cuda::CudaDevice::default()
}

// ------ WGPU backend ------ //
#[cfg(feature = "wgpu")]
pub type Backend = burn::backend::Wgpu<f32>;
#[cfg(feature = "wgpu")]
pub type AutodiffBackend = burn::backend::Autodiff<Backend>;
#[cfg(feature = "wgpu")]
pub fn device() -> burn::backend::wgpu::WgpuDevice {
    burn::backend::wgpu::WgpuDevice::default()
}
