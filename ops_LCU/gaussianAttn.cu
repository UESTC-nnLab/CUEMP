#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#define BLOCK 16

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void gaussianMask_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> means,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> covs,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume1,
    int r)
{
  // batch index
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;

    const int h1 = volume.size(1);
    const int w1 = volume.size(2);
    const int h2 = volume.size(3);
    const int w2 = volume.size(4);

    if (!within_bounds(y, x, h1, w1)) {
        return;
    }

    float mean_x = means[n][y][x][0];
    float mean_y = means[n][y][x][1];
    float cov_1 = covs[n][y][x][0];
    float cov_2 = covs[n][y][x][1];
   
    int rd = 2*r + 1;

  for (int i=0; i<rd; i++) {
    for (int j=0; j<rd; j++) {
       
        int meanCenter_x = static_cast<int>(floor(mean_x));
        int meanCenter_y = static_cast<int>(floor(mean_y));
  
        int x1 = meanCenter_x - r + i;
  
        int y1 = meanCenter_y - r + j;
      if (within_bounds(y1, x1, h2, w2)) {
            float temp1 = (x1-mean_x)/cov_1;
            float temp2 = (y1-mean_y)/cov_2;
            float f1 = -0.5*(temp1*(x1-mean_x)+temp2*(y1-mean_y));
            float exp_comp = exp(f1);
            // float denominator =  6.28*sqrt(cov_1*cov_2);
            // float pdf =  exp_comp/denominator; 
            volume1[n][y][x][y1][x1] =  volume[n][y][x][y1][x1]*3*exp_comp;
      }
    }
  }
}



template <typename scalar_t>
__global__ void gaussianMask_kernel_backward(
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> means,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> covs,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> means_grad,
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> covs_grad,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume1_grad,
    int r)
{
  // batch index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int h1 = volume1_grad.size(1);
  const int w1 = volume1_grad.size(2);
  const int h2 = volume1_grad.size(3);
  const int w2 = volume1_grad.size(4);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

    float mean_x = means[n][y][x][0];
    float mean_y = means[n][y][x][1];
    float cov_1 = covs[n][y][x][0];
    float cov_2 = covs[n][y][x][1];
    int rd = 2*r + 1;

  for (int i=0; i<rd; i++) {
    for (int j=0; j<rd; j++) {
      
      int meanCenter_x = static_cast<int>(floor(mean_x));
      int meanCenter_y = static_cast<int>(floor(mean_y));

      int x1 = meanCenter_x - r + i;

      int y1 = meanCenter_y - r + j;
    if (within_bounds(y1, x1, h2, w2)) {
          float temp1 = (x1-mean_x)/cov_1;
          float temp2 = (y1-mean_y)/cov_2;
          float f1 = -0.5*(temp1*(x1-mean_x)+temp2*(y1-mean_y));
          float exp_comp = exp(f1);

          means_grad[n][y][x][0] += 3*volume[n][y][x][y1][x1]*(exp_comp*(x1-mean_x)/cov_1)*volume1_grad[n][y][x][y1][x1]; //x
          means_grad[n][y][x][1] += 3*volume[n][y][x][y1][x1]*(exp_comp*(y1-mean_y)/cov_2)*volume1_grad[n][y][x][y1][x1]; //y

          float dEXP_cov1 = exp_comp*0.5*(x1-mean_x)*(x1-mean_x)/(cov_1*cov_1);
          // float dP_cov1 = (dEXP_cov1*6.28*sqrt(cov_1*cov_2)-exp_comp*3.14*sqrt(cov_2/cov_1))/(6.28*6.28*cov_1*cov_2);
          float dEXP_cov2 = exp_comp*0.5*(y1-mean_y)*(y1-mean_y)/(cov_2*cov_2);
          // float dP_cov2 = (dEXP_cov2*6.28*sqrt(cov_1*cov_2)-exp_comp*3.14*sqrt(cov_1/cov_2))/(6.28*6.28*cov_1*cov_2);

          covs_grad[n][y][x][0] += (3*volume[n][y][x][y1][x1]*dEXP_cov1)*volume1_grad[n][y][x][y1][x1]; //x
          covs_grad[n][y][x][1] += (3*volume[n][y][x][y1][x1]*dEXP_cov2)*volume1_grad[n][y][x][y1][x1]; //y
    }

    }
  }
}


std::vector<torch::Tensor> gaussianMask_cuda(
    torch::Tensor means,
    torch::Tensor covs,
    torch::Tensor volume,
    int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK, BLOCK);
  
  auto volume1 = torch::zeros_like(volume);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "gaussianMask_kernel", ([&] {
    gaussianMask_kernel<scalar_t><<<blocks, threads>>>(
      means.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      covs.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      volume1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {volume1};

}

std::vector<torch::Tensor> gaussianMask_cuda_backward(
  torch::Tensor means,
  torch::Tensor covs,
  torch::Tensor volume,
  torch::Tensor volume1_grad,
  int radius)
{
const auto batch_size = volume.size(0);
const auto ht = volume.size(1);
const auto wd = volume.size(2);

auto means_grad = torch::zeros_like(means);
auto covs_grad = torch::zeros_like(covs);

const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                  (ht + BLOCK - 1) / BLOCK, 
                  batch_size);

const dim3 threads(BLOCK, BLOCK);



AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "gaussianMask_kernel_backward", ([&] {
  gaussianMask_kernel_backward<scalar_t><<<blocks, threads>>>(
    means.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    covs.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    means_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    covs_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    volume1_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    radius);
 }));

return {means_grad,covs_grad};

}