#include "nerf.h"

#include "json/json.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <mutex>


namespace nerf::context {} // namespace nerf::context
namespace nerf::sampler {} // namespace nerf::sampler
namespace nerf::encoder {} // namespace nerf::encoder
namespace nerf::network {} // namespace nerf::network
namespace nerf::train {} // namespace nerf::train
