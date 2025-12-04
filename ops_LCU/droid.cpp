#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> defCorr_gaussian_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor offset,
  int radius);

std::vector<torch::Tensor> defCorr_gaussian_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor offset,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);
std::vector<torch::Tensor> lowMem_defSample_cuda(
   torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor offset,
  int radius);

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// c++ python binding

std::vector<torch::Tensor> defCorr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor offset,
    torch::Tensor sigma_map,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(offset);
  CHECK_INPUT(sigma_map);

  return defCorr_gaussian_index_cuda_forward(volume, coords, offset, sigma_map, radius);
}

std::vector<torch::Tensor> defCorr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor offset,
    torch::Tensor sigma_map,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(offset);
  CHECK_INPUT(sigma_map);
  CHECK_INPUT(corr_grad);

  return defCorr_gaussian_index_cuda_backward(volume, coords, offset, corr_grad, radius);
}

std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return corr_index_cuda_backward(volume, coords, corr_grad, radius);
}
std::vector<torch::Tensor> gaussianMask(
    torch::Tensor means,
    torch::Tensor covs,
    torch::Tensor volume,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(means);
  CHECK_INPUT(covs);

  return gaussianMask_cuda(means, covs, volume, radius);
}

std::vector<torch::Tensor> gaussianMask_backward(
    torch::Tensor means,
    torch::Tensor covs,
    torch::Tensor volume,
    torch::Tensor volume_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(means);
  CHECK_INPUT(covs);
  CHECK_INPUT(volume_grad);
  return gaussianMask_cuda_backward(means, covs, volume, volume_grad, radius);
}
std::vector<torch::Tensor> lowMem_defSample(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor offset,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(offset);

  return lowMem_defSample_cuda(fmap1, fmap2, coords, offset, radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gaussianMask", &gaussianMask, "gaussianMask kernel");
  m.def("gaussianMask_backward", &gaussianMask_backward, "gaussianMask kernel backward");
  m.def("lowMem_defSample", &lowMem_defSample, "lowMem_defSample kernel");
  // correlation volume kernels
  m.def("corr_index_forward", &corr_index_forward, "corr INDEX forward");
 m.def("corr_index_backward", &corr_index_backward, "corr INDEX backward");
 m.def("defCorr_index_forward", &defCorr_index_forward, "defCorr INDEX forward");
 m.def("defCorr_index_backward", &defCorr_index_backward, "defCorr INDEX backward");
}
