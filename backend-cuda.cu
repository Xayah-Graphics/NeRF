#include "nerf.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <limits>
#include <memory>
#include <mma.h>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>
#include <vector_types.h>

#include "json/json.hpp"

static_assert(std::endian::native == std::endian::little);

namespace nerf::compile_time {
    // These values define the model shape and WMMA kernel layout. They are compile-time
    // constraints, so they stay constexpr but live in one place instead of leaking across the file.
    inline constexpr std::uint32_t kPosFreqs                = 10u;
    inline constexpr std::uint32_t kDirFreqs                = 4u;
    inline constexpr std::uint32_t kPtsInDim                = 3u + 2u * 3u * kPosFreqs;
    inline constexpr std::uint32_t kDirInDim                = 3u + 2u * 3u * kDirFreqs;
    inline constexpr std::uint32_t kDensityInputDim         = 64u;
    inline constexpr std::uint32_t kDensityOutputDim        = 16u;
    inline constexpr std::uint32_t kDensityWidth            = 128u;
    inline constexpr std::uint32_t kDensityHiddenLayers     = 5u;
    inline constexpr std::uint32_t kGeoFeatureDim           = 15u;
    inline constexpr std::uint32_t kColorInputDim           = 48u;
    inline constexpr std::uint32_t kColorOutputDim          = 3u;
    inline constexpr std::uint32_t kColorOutputPaddedDim    = 16u;
    inline constexpr std::uint32_t kColorWidth              = 128u;
    inline constexpr std::uint32_t kColorHiddenLayers       = 3u;
    inline constexpr std::uint32_t kNetworkBatchGranularity = 256u;

    inline constexpr std::uint32_t kFusedWidth             = 128u;
    inline constexpr std::uint32_t kFusedOutputWidth       = 16u;
    inline constexpr std::uint32_t kFusedSkew              = 8u;
    inline constexpr std::uint32_t kFusedInputSkew         = 8u;
    inline constexpr std::uint32_t kFusedBlockRows         = kFusedWidth / 16u;
    inline constexpr std::uint32_t kFusedIters             = 8u;
    inline constexpr std::uint32_t kFusedBatchQuantum      = 16u * kFusedIters;
    inline constexpr std::size_t kFusedForwardShmemDensity = std::max(sizeof(__half) * (kFusedWidth + 16u) * (kDensityInputDim + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));
    inline constexpr std::size_t kFusedForwardShmemColor   = std::max(sizeof(__half) * (kFusedWidth + 16u) * (kColorInputDim + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));
    inline constexpr std::size_t kFusedBackwardShmem       = sizeof(__half) * 16u * kFusedIters * (kFusedWidth + kFusedSkew);
    inline constexpr std::uint32_t kFusedElemsPerLoad      = kFusedBlockRows * 32u * 8u;
    inline constexpr std::uint32_t kFusedWeightsStride     = kFusedWidth * kFusedWidth;
} // namespace nerf::compile_time

namespace nerf::host {
    struct HostCheckpointData {
        std::vector<float> density_params_f32{};
        std::vector<float> color_params_f32{};
    };
    struct DatasetInfo {
        uint32_t image_count;
        uint32_t image_width;
        uint32_t image_height;
        float fx;
        float fy;
        float cx;
        float cy;
    };
} // namespace nerf::host


namespace nerf::runtime {
    enum OccupancyMode : std::uint32_t {
        kOccupancyModeNone       = 0u,
        kOccupancyModeWarmupInit = 1u,
        kOccupancyModeRefresh    = 2u,
    };
    struct TrainingDeviceState {
        std::uint32_t frame_index      = 0u;
        std::uint32_t optimizer_step   = 0u;
        std::uint32_t train_camera_idx = 0u;
        std::uint32_t occupancy_mode   = kOccupancyModeNone;
        NerfTrainStats stats{};
    };
} // namespace nerf::runtime


namespace nerf::sampler {
    struct SampleRay {
        float origin_x             = 0.0f;
        float origin_y             = 0.0f;
        float origin_z             = 0.0f;
        float dir_x                = 0.0f;
        float dir_y                = 0.0f;
        float dir_z                = 0.0f;
        float t_near               = 0.0f;
        float t_far                = 0.0f;
        float cone_angle           = 0.0f;
        float pixel_x              = 0.0f;
        float pixel_y              = 0.0f;
        float gt_r                 = 0.0f;
        float gt_g                 = 0.0f;
        float gt_b                 = 0.0f;
        float gt_a                 = 0.0f;
        std::uint32_t sample_begin = 0u;
        std::uint32_t sample_count = 0u;
    };
    struct SampleStep {
        float x  = 0.0f;
        float y  = 0.0f;
        float z  = 0.0f;
        float dt = 0.0f;
        float dx = 0.0f;
        float dy = 0.0f;
        float dz = 0.0f;
    };
    struct SampleBatchState {
        std::uint32_t active_ray_count  = 0u;
        std::uint32_t sample_step_count = 0u;
    };
    struct SamplerRequest {
        cudaStream_t stream                                    = nullptr;
        const nerf::runtime::TrainingDeviceState* device_state = nullptr;
        const float* cams_4x4_packed                           = nullptr;
        const std::uint8_t* images                             = nullptr;
        const std::uint32_t* bitfield                          = nullptr;
        SampleRay* sample_rays                                 = nullptr;
        SampleStep* sample_steps                               = nullptr;
        SampleBatchState* batch_state                          = nullptr;
        std::uint32_t occupancy_grid_res                       = 0u;
        std::uint32_t rays_per_batch                           = 0u;
        std::uint32_t max_sample_steps_per_ray                 = 0u;
        std::uint32_t image_width                              = 0u;
        std::uint32_t image_height                             = 0u;
        float fx                                               = 0.0f;
        float fy                                               = 0.0f;
        float cx                                               = 0.0f;
        float cy                                               = 0.0f;
        float3 aabb_min{};
        float3 aabb_max{};
    };
} // namespace nerf::sampler


namespace nerf::network {
    using namespace nerf::compile_time;

    template <typename T>
    struct DeviceBuffer {
        T* ptr              = nullptr;
        std::uint64_t count = 0u;
        std::size_t bytes   = 0u;
    };
    struct AdamStepScalars {
        float learning_rate        = 0.0f;
        float grad_scale           = 1.0f;
        float inv_bias_correction1 = 1.0f;
        float inv_bias_correction2 = 1.0f;
        float inv_loss_scale       = 1.0f;
        std::uint32_t skip_update  = 0u;
    };
    struct NetworkWorkspace {
        std::uint32_t rows_capacity        = 0u;
        unsigned char* arena               = nullptr;
        float* inputs_tmp                  = nullptr;
        float* enc_pts                     = nullptr;
        float* enc_dir                     = nullptr;
        float* raw_rgb                     = nullptr;
        float* d_rgb                       = nullptr;
        float* raw_sigma                   = nullptr;
        float* d_sigma                     = nullptr;
        float* trans_tmp                   = nullptr;
        float* loss_sum                    = nullptr;
        float* grad_sumsq                  = nullptr;
        std::uint32_t* nonfinite_flag      = nullptr;
        std::uint32_t* ray_counts_tmp      = nullptr;
        AdamStepScalars* adam_step_scalars = nullptr;
        __half* density_input              = nullptr;
        __half* density_output             = nullptr;
        __half* density_doutput            = nullptr;
        __half* density_forward_hidden     = nullptr;
        __half* density_backward_hidden    = nullptr;
        __half* color_input                = nullptr;
        __half* color_output               = nullptr;
        __half* color_doutput              = nullptr;
        __half* color_dinput               = nullptr;
        __half* color_forward_hidden       = nullptr;
        __half* color_backward_hidden      = nullptr;
    };
    struct FusedNetworkState {
        std::uint32_t input_width    = 0u;
        std::uint32_t width          = 0u;
        std::uint32_t output_width   = 0u;
        std::uint32_t hidden_matmuls = 0u;
        DeviceBuffer<float> params_f32{};
        DeviceBuffer<__half> params{};
        DeviceBuffer<__half> gradients{};
        DeviceBuffer<__half> gradients_tmp{};
        DeviceBuffer<float> adam_m{};
        DeviceBuffer<float> adam_v{};
    };
    struct NetworkSet {
        void* blas_handle                = nullptr;
        void* blas_workspace             = nullptr;
        std::size_t blas_workspace_bytes = 0u;
        FusedNetworkState density{};
        FusedNetworkState color{};
    };
    struct NetworkInferenceRequest {
        const float* encoded_pts = nullptr;
        const float* encoded_dir = nullptr;
        std::uint32_t rows       = 0u;
        float* raw_rgb           = nullptr;
        float* raw_sigma         = nullptr;
    };
    struct NetworkTrainingRequest {
        const nerf::sampler::SampleRay* sample_rays        = nullptr;
        const nerf::sampler::SampleStep* sample_steps      = nullptr;
        const nerf::sampler::SampleBatchState* batch_state = nullptr;
        std::uint32_t max_ray_count                        = 0u;
        std::uint32_t max_sample_count                     = 0u;
    };
    struct NetworkCheckpointTensorLayout {
        char name[64]{};
        std::uint64_t offset        = 0u;
        std::uint32_t rows          = 0u;
        std::uint32_t cols          = 0u;
        std::uint32_t network_index = 0u;
    };
    struct NetworkCheckpointLayout {
        NetworkCheckpointTensorLayout tensors[32]{};
        std::uint32_t tensor_count          = 0u;
        std::uint64_t density_param_count   = 0u;
        std::uint64_t color_param_count     = 0u;
        std::uint32_t density_input_width   = 0u;
        std::uint32_t density_width         = 0u;
        std::uint32_t density_hidden_layers = 0u;
        std::uint32_t density_output_width  = 0u;
        std::uint32_t color_input_width     = 0u;
        std::uint32_t color_width           = 0u;
        std::uint32_t color_hidden_layers   = 0u;
        std::uint32_t color_output_width    = 0u;
    };
} // namespace nerf::network


namespace nerf::encoder {
    using namespace nerf::compile_time;

    namespace {
        __device__ __forceinline__ void encode_position_with_freq(const float px, const float py, const float pz, float* __restrict__ out, const std::uint32_t max_freqs) {
            out[0]               = px;
            out[1]               = py;
            out[2]               = pz;
            std::uint32_t offset = 3u;
            float freq           = 1.0f;
            for (std::uint32_t l = 0u; l < max_freqs; ++l) {
                const float ax = px * freq;
                const float ay = py * freq;
                const float az = pz * freq;
                float sx = 0.0f, cx = 0.0f;
                float sy = 0.0f, cy = 0.0f;
                float sz = 0.0f, cz = 0.0f;
                __sincosf(ax, &sx, &cx);
                __sincosf(ay, &sy, &cy);
                __sincosf(az, &sz, &cz);
                out[offset + 0u] = sx;
                out[offset + 1u] = sy;
                out[offset + 2u] = sz;
                out[offset + 3u] = cx;
                out[offset + 4u] = cy;
                out[offset + 5u] = cz;
                offset += 6u;
                freq *= 2.0f;
            }
        }
        __global__ void k_encode_sample_inputs(const float* __restrict__ inputs, const std::uint32_t rows, float* __restrict__ enc_pts, float* __restrict__ enc_dir) {
            const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= rows) return;
            const std::uint32_t in_base = idx * 7u;
            const float px              = inputs[in_base + 0u];
            const float py              = inputs[in_base + 1u];
            const float pz              = inputs[in_base + 2u];
            const float dx              = inputs[in_base + 4u];
            const float dy              = inputs[in_base + 5u];
            const float dz              = inputs[in_base + 6u];
            float* out_p                = enc_pts + static_cast<std::uint64_t>(idx) * kPtsInDim;
            float* out_d                = enc_dir + static_cast<std::uint64_t>(idx) * kDirInDim;
            encode_position_with_freq(px, py, pz, out_p, kPosFreqs);
            encode_position_with_freq(dx, dy, dz, out_d, kDirFreqs);
        }
        __global__ void k_encode_positions_only(const float* __restrict__ inputs, const std::uint32_t rows, float* __restrict__ enc_pts) {
            const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= rows) return;
            const std::uint32_t in_base = idx * 7u;
            const float px              = inputs[in_base + 0u];
            const float py              = inputs[in_base + 1u];
            const float pz              = inputs[in_base + 2u];
            float* out_p                = enc_pts + static_cast<std::uint64_t>(idx) * kPtsInDim;
            encode_position_with_freq(px, py, pz, out_p, kPosFreqs);
        }
    } // namespace
    bool run_encoder_module(cudaStream_t stream, const float* sample_inputs, const std::uint32_t rows, float* encoded_pts, float* encoded_dir) {
        constexpr std::uint32_t threads = 256u;
        const std::uint32_t blocks  = (rows + threads - 1u) / threads;
        k_encode_sample_inputs<<<blocks, threads, 0, stream>>>(sample_inputs, rows, encoded_pts, encoded_dir);
        return cudaGetLastError() == cudaSuccess;
    }
    bool run_position_encoder_module(cudaStream_t stream, const float* sample_inputs, const std::uint32_t rows, float* encoded_pts) {
        constexpr std::uint32_t threads = 256u;
        const std::uint32_t blocks  = (rows + threads - 1u) / threads;
        k_encode_positions_only<<<blocks, threads, 0, stream>>>(sample_inputs, rows, encoded_pts);
        return cudaGetLastError() == cudaSuccess;
    }
} // namespace nerf::encoder


namespace nerf::sampler {
    namespace {
        __device__ __forceinline__ std::uint32_t hash_u32(std::uint32_t x) {
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return x;
        }
        __device__ __forceinline__ float rand01(const std::uint32_t seed) {
            const std::uint32_t h = hash_u32(seed);
            return static_cast<float>(h & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
        }
        __device__ __forceinline__ float cone_angle_from_focal(const float fx, const float fy) {
            const float inv_fx = fx > 0.0f ? 1.0f / fx : 0.0f;
            const float inv_fy = fy > 0.0f ? 1.0f / fy : 0.0f;
            return 0.5f * (inv_fx + inv_fy);
        }
        __device__ __forceinline__ bool occupied_cell(const std::uint32_t* bitfield, const std::uint32_t grid_res, const float3 aabb_min, const float3 aabb_max, const float3 p) {
            const float size_x = aabb_max.x - aabb_min.x;
            const float size_y = aabb_max.y - aabb_min.y;
            const float size_z = aabb_max.z - aabb_min.z;
            const float3 rel   = make_float3((p.x - aabb_min.x) / size_x, (p.y - aabb_min.y) / size_y, (p.z - aabb_min.z) / size_z);
            if (rel.x < 0.0f || rel.x >= 1.0f || rel.y < 0.0f || rel.y >= 1.0f || rel.z < 0.0f || rel.z >= 1.0f) return false;
            const std::uint32_t ix   = min(grid_res - 1u, static_cast<std::uint32_t>(rel.x * static_cast<float>(grid_res)));
            const std::uint32_t iy   = min(grid_res - 1u, static_cast<std::uint32_t>(rel.y * static_cast<float>(grid_res)));
            const std::uint32_t iz   = min(grid_res - 1u, static_cast<std::uint32_t>(rel.z * static_cast<float>(grid_res)));
            const std::uint32_t idx  = ix + iy * grid_res + iz * grid_res * grid_res;
            const std::uint32_t word = idx >> 5u;
            const std::uint32_t bit  = idx & 31u;
            return (bitfield[word] & 1u << bit) != 0u;
        }
        struct CameraParams {
            float4 row0{};
            float4 row1{};
            float4 row2{};
            float fx = 0.0f;
            float fy = 0.0f;
            float cx = 0.0f;
            float cy = 0.0f;
        };
        __device__ __forceinline__ CameraParams load_camera_params(const float* cams_4x4_packed, const std::uint32_t camera_idx, const float fx, const float fy, const float cx, const float cy) {
            const std::uint32_t cam_base = camera_idx * 16u;
            CameraParams cam{};
            cam.row0 = make_float4(cams_4x4_packed[cam_base + 0u], cams_4x4_packed[cam_base + 1u], cams_4x4_packed[cam_base + 2u], cams_4x4_packed[cam_base + 3u]);
            cam.row1 = make_float4(cams_4x4_packed[cam_base + 4u], cams_4x4_packed[cam_base + 5u], cams_4x4_packed[cam_base + 6u], cams_4x4_packed[cam_base + 7u]);
            cam.row2 = make_float4(cams_4x4_packed[cam_base + 8u], cams_4x4_packed[cam_base + 9u], cams_4x4_packed[cam_base + 10u], cams_4x4_packed[cam_base + 11u]);
            cam.fx   = fx;
            cam.fy   = fy;
            cam.cx   = cx;
            cam.cy   = cy;
            return cam;
        }
        __device__ __forceinline__ bool compute_world_ray_dir(const CameraParams& cam, const float pixel_x, const float pixel_y_flipped, float3* out_ray_dir) {
            float3 dir_cam{};
            dir_cam.x = (pixel_x - cam.cx) / cam.fx;
            dir_cam.y = (pixel_y_flipped - cam.cy) / cam.fy;
            dir_cam.z = -1.0f;
            float3 ray_dir{};
            ray_dir.x        = cam.row0.x * dir_cam.x + cam.row0.y * dir_cam.y + cam.row0.z * dir_cam.z;
            ray_dir.y        = cam.row1.x * dir_cam.x + cam.row1.y * dir_cam.y + cam.row1.z * dir_cam.z;
            ray_dir.z        = cam.row2.x * dir_cam.x + cam.row2.y * dir_cam.y + cam.row2.z * dir_cam.z;
            const float len2 = ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z;
            if (len2 <= 1e-16f) return false;
            const float inv_len = rsqrtf(len2);
            ray_dir.x *= inv_len;
            ray_dir.y *= inv_len;
            ray_dir.z *= inv_len;
            *out_ray_dir = ray_dir;
            return true;
        }
        __device__ __forceinline__ bool intersect_aabb_ray(const float3 ray_origin, const float3 ray_dir, const float3 aabb_min, const float3 aabb_max, float* out_t_near, float* out_t_far) {
            constexpr float eps = 1e-8f;
            float3 inv_dir{};
            inv_dir.x          = fabsf(ray_dir.x) > eps ? 1.0f / ray_dir.x : ray_dir.x >= 0.0f ? 1e8f : -1e8f;
            inv_dir.y          = fabsf(ray_dir.y) > eps ? 1.0f / ray_dir.y : ray_dir.y >= 0.0f ? 1e8f : -1e8f;
            inv_dir.z          = fabsf(ray_dir.z) > eps ? 1.0f / ray_dir.z : ray_dir.z >= 0.0f ? 1e8f : -1e8f;
            const float3 t0    = make_float3((aabb_min.x - ray_origin.x) * inv_dir.x, (aabb_min.y - ray_origin.y) * inv_dir.y, (aabb_min.z - ray_origin.z) * inv_dir.z);
            const float3 t1    = make_float3((aabb_max.x - ray_origin.x) * inv_dir.x, (aabb_max.y - ray_origin.y) * inv_dir.y, (aabb_max.z - ray_origin.z) * inv_dir.z);
            const float3 tminv = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
            const float3 tmaxv = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
            float t_near       = fmaxf(fmaxf(tminv.x, tminv.y), tminv.z);
            const float t_far  = fminf(fminf(tmaxv.x, tmaxv.y), tmaxv.z);
            if (!(t_near <= t_far && t_far > 0.0f)) return false;
            if (t_near < 0.0f) t_near = 0.0f;
            *out_t_near = t_near;
            *out_t_far  = t_far;
            return true;
        }
        __device__ __forceinline__ float sigmoid_unit(const float x) {
            return 1.0f / (1.0f + __expf(-x));
        }
        __device__ __forceinline__ float softplus_sigma_raw(const float x) {
            if (x > 20.0f) return x;
            if (x < -20.0f) return __expf(x);
            return log1pf(__expf(x));
        }
        __global__ void k_begin_sampling_step(SampleBatchState* __restrict__ batch_state) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            batch_state->active_ray_count  = 0u;
            batch_state->sample_step_count = 0u;
        }
        __global__ void k_sample_rays_flat(const nerf::runtime::TrainingDeviceState* __restrict__ device_state, const float* __restrict__ cams_4x4_packed, const std::uint8_t* __restrict__ images, const std::uint32_t image_width, const std::uint32_t image_height, const float fx, const float fy, const float cx, const float cy, const std::uint32_t rays_per_batch, const std::uint32_t max_sample_steps_per_ray, const float3 aabb_min, const float3 aabb_max,
            const std::uint32_t* __restrict__ bitfield, const std::uint32_t grid_res, SampleRay* __restrict__ sample_rays, SampleStep* __restrict__ sample_steps, SampleBatchState* __restrict__ batch_state) {
            const std::uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray_idx >= rays_per_batch) return;
            __shared__ CameraParams shared_cam;
            __shared__ std::uint32_t shared_frame_index;
            __shared__ std::uint32_t shared_camera_idx;
            __shared__ const std::uint8_t* shared_camera_image;
            const std::uint64_t per_cam_bytes = static_cast<std::uint64_t>(image_width) * static_cast<std::uint64_t>(image_height) * 4ull;
            if (threadIdx.x == 0u) {
                shared_frame_index  = device_state->frame_index;
                shared_camera_idx   = device_state->train_camera_idx;
                shared_cam          = load_camera_params(cams_4x4_packed, shared_camera_idx, fx, fy, cx, cy);
                shared_camera_image = images + static_cast<std::uint64_t>(shared_camera_idx) * per_cam_bytes;
            }
            __syncthreads();
            const std::uint32_t seed    = shared_frame_index * 1315423911u ^ ray_idx * 9781u;
            const float pixel_x         = rand01(seed ^ 0xA511E9B3u) * static_cast<float>(image_width);
            const float pixel_y         = rand01(seed ^ 0x63D83595u) * static_cast<float>(image_height);
            const float pixel_y_flipped = static_cast<float>(image_height) - 1.0f - pixel_y;
            const float3 ray_origin     = make_float3(shared_cam.row0.w, shared_cam.row1.w, shared_cam.row2.w);
            float3 ray_dir{};
            float t_near = 0.0f;
            float t_far  = 0.0f;
            if (!compute_world_ray_dir(shared_cam, pixel_x, pixel_y_flipped, &ray_dir)) return;
            if (!intersect_aabb_ray(ray_origin, ray_dir, aabb_min, aabb_max, &t_near, &t_far)) return;
            const float t_range = t_far - t_near;
            if (!(t_range > 0.0f)) return;
            const float dt_min     = t_range / static_cast<float>(max_sample_steps_per_ray);
            const float cone_angle = cone_angle_from_focal(shared_cam.fx, shared_cam.fy);
            float sample_t_mid[256];
            float sample_dt[256];
            std::uint32_t sample_count = 0u;
            std::uint32_t march_i      = 0u;
            float t                    = t_near;
            while (t < t_far && march_i < max_sample_steps_per_ray) {
                const float dt_candidate = fmaxf(dt_min, t * cone_angle);
                const float dt           = fminf(dt_candidate, t_far - t);
                const float t_mid        = t + 0.5f * dt;
                const float3 pos         = make_float3(ray_origin.x + t_mid * ray_dir.x, ray_origin.y + t_mid * ray_dir.y, ray_origin.z + t_mid * ray_dir.z);
                if (occupied_cell(bitfield, grid_res, aabb_min, aabb_max, pos)) {
                    sample_t_mid[sample_count] = t_mid;
                    sample_dt[sample_count]    = dt;
                    ++sample_count;
                }
                t += dt;
                ++march_i;
            }
            if (sample_count == 0u) return;
            const float px                   = fmaxf(0.0f, fminf(pixel_x, static_cast<float>(image_width) - 1.0f));
            const float py                   = fmaxf(0.0f, fminf(pixel_y, static_cast<float>(image_height) - 1.0f));
            const int x0                     = __float2int_rd(px);
            const int y0                     = __float2int_rd(py);
            const int x1                     = min(x0 + 1, static_cast<int>(image_width) - 1);
            const int y1                     = min(y0 + 1, static_cast<int>(image_height) - 1);
            const float tx                   = px - static_cast<float>(x0);
            const float ty                   = py - static_cast<float>(y0);
            const int stride                 = static_cast<int>(image_width) * 4;
            const int idx00                  = y0 * stride + x0 * 4;
            const int idx10                  = y0 * stride + x1 * 4;
            const int idx01                  = y1 * stride + x0 * 4;
            const int idx11                  = y1 * stride + x1 * 4;
            constexpr float inv255               = 1.0f / 255.0f;
            const float3 s00                 = make_float3(static_cast<float>(shared_camera_image[idx00 + 0]) * inv255, static_cast<float>(shared_camera_image[idx00 + 1]) * inv255, static_cast<float>(shared_camera_image[idx00 + 2]) * inv255);
            const float3 s10                 = make_float3(static_cast<float>(shared_camera_image[idx10 + 0]) * inv255, static_cast<float>(shared_camera_image[idx10 + 1]) * inv255, static_cast<float>(shared_camera_image[idx10 + 2]) * inv255);
            const float3 s01                 = make_float3(static_cast<float>(shared_camera_image[idx01 + 0]) * inv255, static_cast<float>(shared_camera_image[idx01 + 1]) * inv255, static_cast<float>(shared_camera_image[idx01 + 2]) * inv255);
            const float3 s11                 = make_float3(static_cast<float>(shared_camera_image[idx11 + 0]) * inv255, static_cast<float>(shared_camera_image[idx11 + 1]) * inv255, static_cast<float>(shared_camera_image[idx11 + 2]) * inv255);
            const float a00                  = static_cast<float>(shared_camera_image[idx00 + 3]) * inv255;
            const float a10                  = static_cast<float>(shared_camera_image[idx10 + 3]) * inv255;
            const float a01                  = static_cast<float>(shared_camera_image[idx01 + 3]) * inv255;
            const float a11                  = static_cast<float>(shared_camera_image[idx11 + 3]) * inv255;
            const float3 a                   = make_float3(s00.x * (1.0f - tx) + s10.x * tx, s00.y * (1.0f - tx) + s10.y * tx, s00.z * (1.0f - tx) + s10.z * tx);
            const float3 b                   = make_float3(s01.x * (1.0f - tx) + s11.x * tx, s01.y * (1.0f - tx) + s11.y * tx, s01.z * (1.0f - tx) + s11.z * tx);
            const float3 gt_rgb              = make_float3(a.x * (1.0f - ty) + b.x * ty, a.y * (1.0f - ty) + b.y * ty, a.z * (1.0f - ty) + b.z * ty);
            const float aa                   = a00 * (1.0f - tx) + a10 * tx;
            const float ab                   = a01 * (1.0f - tx) + a11 * tx;
            const float alpha                = aa * (1.0f - ty) + ab * ty;
            const float3 gt                  = make_float3(gt_rgb.x * alpha + (1.0f - alpha), gt_rgb.y * alpha + (1.0f - alpha), gt_rgb.z * alpha + (1.0f - alpha));
            const std::uint32_t active_slot  = atomicAdd(&batch_state->active_ray_count, 1u);
            const std::uint32_t sample_begin = atomicAdd(&batch_state->sample_step_count, sample_count);
            for (std::uint32_t i = 0u; i < sample_count; ++i) {
                const float t_mid = sample_t_mid[i];
                SampleStep step{};
                step.x                         = ray_origin.x + t_mid * ray_dir.x;
                step.y                         = ray_origin.y + t_mid * ray_dir.y;
                step.z                         = ray_origin.z + t_mid * ray_dir.z;
                step.dt                        = sample_dt[i];
                step.dx                        = ray_dir.x;
                step.dy                        = ray_dir.y;
                step.dz                        = ray_dir.z;
                sample_steps[sample_begin + i] = step;
            }
            SampleRay sample_ray{};
            sample_ray.origin_x      = ray_origin.x;
            sample_ray.origin_y      = ray_origin.y;
            sample_ray.origin_z      = ray_origin.z;
            sample_ray.dir_x         = ray_dir.x;
            sample_ray.dir_y         = ray_dir.y;
            sample_ray.dir_z         = ray_dir.z;
            sample_ray.t_near        = t_near;
            sample_ray.t_far         = t_far;
            sample_ray.cone_angle    = cone_angle;
            sample_ray.pixel_x       = pixel_x;
            sample_ray.pixel_y       = pixel_y;
            sample_ray.gt_r          = gt.x;
            sample_ray.gt_g          = gt.y;
            sample_ray.gt_b          = gt.z;
            sample_ray.gt_a          = alpha;
            sample_ray.sample_begin  = sample_begin;
            sample_ray.sample_count  = sample_count;
            sample_rays[active_slot] = sample_ray;
        }
        __global__ void k_generate_inference_inputs(const float* __restrict__ cams_4x4_packed, const std::uint32_t camera_idx, const std::uint32_t image_width, const std::uint32_t image_height, const float fx, const float fy, const float cx, const float cy, const std::uint32_t ray_start, const std::uint32_t ray_count, const std::uint32_t samples_per_ray, const float3 aabb_min, const float3 aabb_max, float* __restrict__ out_inputs, std::uint32_t* __restrict__ out_ray_counts) {
            const std::uint32_t local_ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (local_ray >= ray_count) return;

            __shared__ CameraParams shared_cam;
            if (threadIdx.x == 0u) shared_cam = load_camera_params(cams_4x4_packed, camera_idx, fx, fy, cx, cy);
            __syncthreads();

            const std::uint32_t global_ray = ray_start + local_ray;
            const std::uint32_t px         = global_ray % image_width;
            const std::uint32_t py         = global_ray / image_width;
            const std::uint64_t ray_base   = static_cast<std::uint64_t>(local_ray) * samples_per_ray;
            if (py >= image_height) {
                out_ray_counts[local_ray] = 0u;
                for (std::uint32_t i = 0u; i < samples_per_ray; ++i) {
                    const std::uint64_t base = (ray_base + i) * 7ull;
                    out_inputs[base + 0u]    = 0.0f;
                    out_inputs[base + 1u]    = 0.0f;
                    out_inputs[base + 2u]    = 0.0f;
                    out_inputs[base + 3u]    = 0.0f;
                    out_inputs[base + 4u]    = 0.0f;
                    out_inputs[base + 5u]    = 0.0f;
                    out_inputs[base + 6u]    = 0.0f;
                }
                return;
            }

            const float pixel_x         = static_cast<float>(px) + 0.5f;
            const float pixel_y         = static_cast<float>(py) + 0.5f;
            const float pixel_y_flipped = static_cast<float>(image_height) - 1.0f - pixel_y;

            const float3 ray_origin = make_float3(shared_cam.row0.w, shared_cam.row1.w, shared_cam.row2.w);
            float3 ray_dir{};
            float t_near        = 0.0f;
            float t_far         = 0.0f;
            const bool hit_aabb = compute_world_ray_dir(shared_cam, pixel_x, pixel_y_flipped, &ray_dir) && intersect_aabb_ray(ray_origin, ray_dir, aabb_min, aabb_max, &t_near, &t_far);
            const float dt      = hit_aabb ? (t_far - t_near) / static_cast<float>(samples_per_ray) : 0.0f;

            if (!(dt > 0.0f)) {
                out_ray_counts[local_ray] = 0u;
                for (std::uint32_t i = 0u; i < samples_per_ray; ++i) {
                    const std::uint64_t base = (ray_base + i) * 7ull;
                    out_inputs[base + 0u]    = 0.0f;
                    out_inputs[base + 1u]    = 0.0f;
                    out_inputs[base + 2u]    = 0.0f;
                    out_inputs[base + 3u]    = 0.0f;
                    out_inputs[base + 4u]    = 0.0f;
                    out_inputs[base + 5u]    = 0.0f;
                    out_inputs[base + 6u]    = 0.0f;
                }
                return;
            }

            out_ray_counts[local_ray] = samples_per_ray;
            for (std::uint32_t i = 0u; i < samples_per_ray; ++i) {
                const float t_mid        = t_near + (static_cast<float>(i) + 0.5f) * dt;
                const std::uint64_t s    = ray_base + i;
                const std::uint64_t base = s * 7ull;
                out_inputs[base + 0u]    = ray_origin.x + t_mid * ray_dir.x;
                out_inputs[base + 1u]    = ray_origin.y + t_mid * ray_dir.y;
                out_inputs[base + 2u]    = ray_origin.z + t_mid * ray_dir.z;
                out_inputs[base + 3u]    = dt;
                out_inputs[base + 4u]    = ray_dir.x;
                out_inputs[base + 5u]    = ray_dir.y;
                out_inputs[base + 6u]    = ray_dir.z;
            }
        }
        __global__ void k_composite_inference_rgba8(const float* __restrict__ raw_rgb, const float* __restrict__ raw_sigma, const float* __restrict__ inputs, const std::uint32_t* __restrict__ ray_counts, const std::uint32_t ray_start, const std::uint32_t ray_count, const std::uint32_t samples_per_ray, const std::uint32_t image_width, const std::uint32_t image_height, std::uint32_t* __restrict__ out_rgba) {
            const std::uint32_t local_ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (local_ray >= ray_count) return;

            const std::uint32_t global_ray = ray_start + local_ray;
            const std::uint32_t py         = global_ray / image_width;
            if (py >= image_height) return;

            const std::uint32_t count    = ray_counts[local_ray];
            const std::uint64_t ray_base = static_cast<std::uint64_t>(local_ray) * samples_per_ray;
            float transmittance          = 1.0f;
            float accum_r                = 0.0f;
            float accum_g                = 0.0f;
            float accum_b                = 0.0f;

            for (std::uint32_t i = 0u; i < count; ++i) {
                const std::uint64_t sample_index = ray_base + i;
                const float dt                   = fmaxf(0.0f, inputs[sample_index * 7ull + 3ull]);
                const float sigma                = softplus_sigma_raw(raw_sigma[sample_index]);
                const float alpha                = 1.0f - __expf(-(sigma * dt));
                const float weight               = transmittance * alpha;
                accum_r += weight * sigmoid_unit(raw_rgb[sample_index * 3ull + 0ull]);
                accum_g += weight * sigmoid_unit(raw_rgb[sample_index * 3ull + 1ull]);
                accum_b += weight * sigmoid_unit(raw_rgb[sample_index * 3ull + 2ull]);
                transmittance *= (1.0f - alpha);
            }

            const float r          = fminf(1.0f, fmaxf(0.0f, accum_r + transmittance));
            const float g          = fminf(1.0f, fmaxf(0.0f, accum_g + transmittance));
            const float b          = fminf(1.0f, fmaxf(0.0f, accum_b + transmittance));
            const std::uint32_t ur = __float2uint_rn(r * 255.0f);
            const std::uint32_t ug = __float2uint_rn(g * 255.0f);
            const std::uint32_t ub = __float2uint_rn(b * 255.0f);
            out_rgba[global_ray]   = (255u << 24u) | (ub << 16u) | (ug << 8u) | ur;
        }
    } // namespace
    bool run_sampler(const SamplerRequest& request) {
        if (!request.stream || !request.device_state || !request.cams_4x4_packed || !request.images || !request.bitfield) return false;
        if (!request.sample_rays || !request.sample_steps || !request.batch_state) return false;
        if (request.rays_per_batch == 0u || request.max_sample_steps_per_ray == 0u) return false;
        k_begin_sampling_step<<<1, 1, 0, request.stream>>>(request.batch_state);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return false;
        const std::uint32_t blocks = (request.rays_per_batch + 128u - 1u) / 128u;
        k_sample_rays_flat<<<blocks, 128u, 0, request.stream>>>(request.device_state, request.cams_4x4_packed, request.images, request.image_width, request.image_height, request.fx, request.fy, request.cx, request.cy, request.rays_per_batch, request.max_sample_steps_per_ray, request.aabb_min, request.aabb_max, request.bitfield, request.occupancy_grid_res, request.sample_rays, request.sample_steps, request.batch_state);
        return cudaGetLastError() == cudaSuccess;
    }
    bool write_inference_inputs(cudaStream_t stream, const float* cams_4x4_packed, const std::uint32_t camera_idx, const std::uint32_t image_width, const std::uint32_t image_height, const float fx, const float fy, const float cx, const float cy, const std::uint32_t ray_start, const std::uint32_t ray_count, const std::uint32_t samples_per_ray, const float3 aabb_min, const float3 aabb_max, float* out_inputs, std::uint32_t* out_ray_counts) {
        if (ray_count == 0u) return true;
        constexpr std::uint32_t inference_threads = 256u;
        const std::uint32_t inference_blocks  = (ray_count + inference_threads - 1u) / inference_threads;
        k_generate_inference_inputs<<<inference_blocks, inference_threads, 0, stream>>>(cams_4x4_packed, camera_idx, image_width, image_height, fx, fy, cx, cy, ray_start, ray_count, samples_per_ray, aabb_min, aabb_max, out_inputs, out_ray_counts);
        return cudaGetLastError() == cudaSuccess;
    }
    bool write_inference_rgba(cudaStream_t stream, const float* raw_rgb, const float* raw_sigma, const float* inputs, const std::uint32_t* ray_counts, const std::uint32_t ray_start, const std::uint32_t ray_count, const std::uint32_t samples_per_ray, const std::uint32_t image_width, const std::uint32_t image_height, std::uint32_t* out_rgba) {
        if (ray_count == 0u) return true;
        constexpr std::uint32_t inference_threads = 256u;
        const std::uint32_t inference_blocks  = (ray_count + inference_threads - 1u) / inference_threads;
        k_composite_inference_rgba8<<<inference_blocks, inference_threads, 0, stream>>>(raw_rgb, raw_sigma, inputs, ray_counts, ray_start, ray_count, samples_per_ray, image_width, image_height, out_rgba);
        return cudaGetLastError() == cudaSuccess;
    }
} // namespace nerf::sampler


namespace nerf::network {
    using namespace nerf::compile_time;

    namespace {
        __device__ void fully_fused_hidden_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ out_intermediate_threadblock_this_layer) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[kFusedBlockRows];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[kFusedIters];
            const std::uint32_t lane        = threadIdx.x;
            const std::uint32_t warp        = threadIdx.y;
            const std::uint32_t lane_offset = 8u * lane % kFusedWidth;
            const std::uint32_t row         = (8u * lane + warp * 8u * 32u) / kFusedWidth;
            const std::uint32_t weights_col = 16u * warp;
            __syncthreads();
#pragma unroll
            for (std::uint32_t i = 0u; i < kFusedBlockRows; ++i) {
                nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i + weights_col * kFusedWidth, kFusedWidth);
            }
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                nvcuda::wmma::fill_fragment(result_frag[iter], 0.0f);
#pragma unroll
                for (std::uint32_t i = 0u; i < kFusedBlockRows; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + 16u * iter * (kFusedWidth + kFusedSkew), kFusedWidth + kFusedSkew);
                    nvcuda::wmma::mma_sync(result_frag[iter], act_frag, weights_frag[i], result_frag[iter]);
                }
#pragma unroll
                for (int element = 0; element < result_frag[iter].num_elements; ++element) {
                    result_frag[iter].x[element] = __hgt(result_frag[iter].x[element], __float2half_rn(0.0f)) ? result_frag[iter].x[element] : __float2half_rn(0.0f);
                }
            }
            __syncthreads();
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + 16u * iter * (kFusedWidth + kFusedSkew), result_frag[iter], kFusedWidth + kFusedSkew, nvcuda::wmma::mem_row_major);
            }
            if (out_intermediate_threadblock_this_layer != nullptr) {
                __syncthreads();
#pragma unroll
                for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                    *reinterpret_cast<int4*>(&out_intermediate_threadblock_this_layer[lane_offset + (row + 16u * iter) * kFusedWidth]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * iter) * (kFusedWidth + kFusedSkew)]);
                }
            }
        }
        __device__ void fully_fused_input_forward(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, __half* __restrict__ out_intermediate_threadblock_this_layer, const std::uint32_t input_width) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[kFusedIters];
            const std::uint32_t lane            = threadIdx.x;
            const std::uint32_t warp            = threadIdx.y;
            const std::uint32_t lane_offset     = 8u * lane % kFusedWidth;
            const std::uint32_t row             = (8u * lane + warp * 8u * 32u) / kFusedWidth;
            const std::uint32_t weights_col     = 16u * warp;
            __half* const weights_shmem         = act_shmem + 16u * (input_width + kFusedInputSkew);
            const std::uint32_t thread_elem_idx = (lane + warp * 32u) * 8u;
            constexpr std::uint32_t elems_per_load  = kFusedElemsPerLoad;
            const std::uint32_t n_weight_elems  = kFusedWidth * input_width;
            for (std::uint32_t idx = thread_elem_idx; idx < n_weight_elems; idx += elems_per_load) {
                const std::uint32_t idx_skewed                       = idx + idx / input_width * kFusedInputSkew;
                *reinterpret_cast<int4*>(&weights_shmem[idx_skewed]) = *reinterpret_cast<const int4*>(&weights_this_layer[idx]);
            }
            const std::uint32_t tensor_ops = input_width / 16u;
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                const std::uint32_t input_elems = 16u * input_width;
                for (std::uint32_t idx = thread_elem_idx; idx < input_elems; idx += elems_per_load) {
                    const std::uint32_t idx_skewed                   = idx + idx / input_width * kFusedInputSkew;
                    *reinterpret_cast<int4*>(&act_shmem[idx_skewed]) = *reinterpret_cast<const int4*>(&input_threadblock[iter * input_elems + idx]);
                }
                __syncthreads();
                nvcuda::wmma::fill_fragment(result_frag[iter], 0.0f);
#pragma unroll
                for (std::uint32_t i = 0u; i < tensor_ops; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i, input_width + kFusedInputSkew);
                    nvcuda::wmma::load_matrix_sync(weights_frag, weights_shmem + 16u * i + weights_col * (input_width + kFusedInputSkew), input_width + kFusedInputSkew);
                    nvcuda::wmma::mma_sync(result_frag[iter], act_frag, weights_frag, result_frag[iter]);
                }
                __syncthreads();
#pragma unroll
                for (int element = 0; element < result_frag[iter].num_elements; ++element) {
                    result_frag[iter].x[element] = __hgt(result_frag[iter].x[element], __float2half_rn(0.0f)) ? result_frag[iter].x[element] : __float2half_rn(0.0f);
                }
            }
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + 16u * iter * (kFusedWidth + kFusedSkew), result_frag[iter], kFusedWidth + kFusedSkew, nvcuda::wmma::mem_row_major);
            }
            if (out_intermediate_threadblock_this_layer != nullptr) {
                __syncthreads();
#pragma unroll
                for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                    *reinterpret_cast<int4*>(&out_intermediate_threadblock_this_layer[lane_offset + (row + 16u * iter) * kFusedWidth]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * iter) * (kFusedWidth + kFusedSkew)]);
                }
            }
        }
        __device__ void fully_fused_output_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ out) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[kFusedBlockRows];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag;
            const std::uint32_t lane                                                                         = threadIdx.x;
            const std::uint32_t warp                                                                         = threadIdx.y;
            __half* const weights_shmem                                                                      = act_shmem + kFusedIters * 16u * (kFusedWidth + kFusedSkew);
            const std::uint32_t weights_row                                                                  = 8u * lane % kFusedWidth;
            const std::uint32_t weights_col                                                                  = (8u * lane + 8u * 32u * warp) / kFusedWidth;
            *reinterpret_cast<int4*>(&weights_shmem[weights_row + weights_col * (kFusedWidth + kFusedSkew)]) = *reinterpret_cast<const int4*>(&weights_this_layer[weights_row + weights_col * kFusedWidth]);
            __syncthreads();
#pragma unroll
            for (std::uint32_t i = 0u; i < kFusedBlockRows; ++i) {
                nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16u * i, kFusedWidth + kFusedSkew);
            }
            for (std::uint32_t idx = warp; idx < kFusedIters; idx += kFusedBlockRows) {
                nvcuda::wmma::fill_fragment(result_frag, 0.0f);
#pragma unroll
                for (std::uint32_t i = 0u; i < kFusedBlockRows; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + 16u * idx * (kFusedWidth + kFusedSkew), kFusedWidth + kFusedSkew);
                    nvcuda::wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
                }
                nvcuda::wmma::store_matrix_sync(out + idx * 16u * kFusedOutputWidth, result_frag, kFusedOutputWidth, nvcuda::wmma::mem_row_major);
            }
        }
        __device__ void fully_fused_hidden_backward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ out_intermediate_threadblock_this_layer, const __half* __restrict__ activation_aux) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag[kFusedBlockRows];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[kFusedIters];
            const std::uint32_t lane        = threadIdx.x;
            const std::uint32_t warp        = threadIdx.y;
            const std::uint32_t lane_offset = 8u * lane % kFusedWidth;
            const std::uint32_t row         = (8u * lane + warp * 8u * 32u) / kFusedWidth;
            const std::uint32_t weights_col = 16u * warp;
            __syncthreads();
#pragma unroll
            for (std::uint32_t i = 0u; i < kFusedBlockRows; ++i) {
                nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i * kFusedWidth + weights_col, kFusedWidth);
            }
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                nvcuda::wmma::fill_fragment(result_frag[iter], 0.0f);
#pragma unroll
                for (std::uint32_t i = 0u; i < kFusedBlockRows; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + 16u * iter * (kFusedWidth + kFusedSkew), kFusedWidth + kFusedSkew);
                    nvcuda::wmma::mma_sync(result_frag[iter], act_frag, weights_frag[i], result_frag[iter]);
                }
                nvcuda::wmma::load_matrix_sync(act_frag, activation_aux + weights_col + iter * 16u * kFusedWidth, kFusedWidth);
#pragma unroll
                for (int element = 0; element < result_frag[iter].num_elements; ++element) {
                    result_frag[iter].x[element] = __hgt(act_frag.x[element], __float2half_rn(0.0f)) ? result_frag[iter].x[element] : __float2half_rn(0.0f);
                }
            }
            __syncthreads();
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + iter * 16u * (kFusedWidth + kFusedSkew), result_frag[iter], kFusedWidth + kFusedSkew, nvcuda::wmma::mem_row_major);
            }
            __syncthreads();
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                *reinterpret_cast<int4*>(&out_intermediate_threadblock_this_layer[lane_offset + (row + 16u * iter) * kFusedWidth]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * iter) * (kFusedWidth + kFusedSkew)]);
            }
        }
        __global__ void k_fully_fused_forward(const __half* __restrict__ input, const __half* __restrict__ weights, __half* __restrict__ out_intermediate, __half* __restrict__ output, const std::uint32_t input_width, const std::uint32_t hidden_matmuls) {
            extern __shared__ __half shmem[];
            __half* const act_shmem      = shmem;
            const std::uint32_t elem_idx = 16u * blockIdx.x * kFusedIters;
            fully_fused_input_forward(act_shmem, input + elem_idx * input_width, weights, out_intermediate ? out_intermediate + elem_idx * kFusedWidth : nullptr, input_width);
            const std::uint32_t first_weights_stride = kFusedWidth * input_width;
            constexpr std::uint32_t weights_stride       = kFusedWeightsStride;
            const std::uint32_t layer_stride         = kFusedWidth * gridDim.x * kFusedBatchQuantum;
            for (std::uint32_t layer = 0u; layer < hidden_matmuls; ++layer) {
                fully_fused_hidden_forward(act_shmem, weights + first_weights_stride + weights_stride * layer, out_intermediate ? out_intermediate + layer_stride * (layer + 1u) + elem_idx * kFusedWidth : nullptr);
            }
            fully_fused_output_forward(act_shmem, weights + first_weights_stride + weights_stride * hidden_matmuls, output + elem_idx * kFusedOutputWidth);
        }
        __global__ void k_fully_fused_backward(const __half* __restrict__ doutput, const __half* __restrict__ weights, __half* __restrict__ out_intermediate, const __half* __restrict__ forward, const std::uint32_t batch_size, const std::uint32_t hidden_matmuls) {
            extern __shared__ __half shmem[];
            __half* const act_shmem = shmem;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[kFusedIters];
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
            const std::uint32_t lane           = threadIdx.x;
            const std::uint32_t warp           = threadIdx.y;
            const std::uint32_t lane_offset    = 8u * lane % kFusedWidth;
            const std::uint32_t row            = (8u * lane + warp * 8u * 32u) / kFusedWidth;
            const std::uint32_t weights_col    = 16u * warp;
            const std::uint32_t elem_idx_base  = 16u * blockIdx.x * kFusedIters;
            constexpr std::uint32_t weights_stride = kFusedWeightsStride;
            const std::uint32_t layer_stride   = kFusedWidth * batch_size;
            nvcuda::wmma::load_matrix_sync(weights_frag, weights + weights_stride * hidden_matmuls + weights_col, kFusedWidth);
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                nvcuda::wmma::fill_fragment(result_frag[iter], 0.0f);
                nvcuda::wmma::load_matrix_sync(act_frag, doutput + (elem_idx_base + 16u * iter) * kFusedOutputWidth, kFusedOutputWidth);
                nvcuda::wmma::mma_sync(result_frag[iter], act_frag, weights_frag, result_frag[iter]);
                nvcuda::wmma::load_matrix_sync(forward_frag, forward + layer_stride * hidden_matmuls + weights_col + (elem_idx_base + iter * 16u) * kFusedWidth, kFusedWidth);
#pragma unroll
                for (int element = 0; element < result_frag[iter].num_elements; ++element) {
                    result_frag[iter].x[element] = __hgt(forward_frag.x[element], __float2half_rn(0.0f)) ? result_frag[iter].x[element] : __float2half_rn(0.0f);
                }
            }
            __syncthreads();
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + 16u * iter * (kFusedWidth + kFusedSkew), result_frag[iter], kFusedWidth + kFusedSkew, nvcuda::wmma::mem_row_major);
            }
            __syncthreads();
#pragma unroll
            for (std::uint32_t iter = 0u; iter < kFusedIters; ++iter) {
                *reinterpret_cast<int4*>(&out_intermediate[lane_offset + (row + elem_idx_base + iter * 16u) * kFusedWidth]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * iter) * (kFusedWidth + kFusedSkew)]);
            }
            for (std::uint32_t layer = 0u; layer < hidden_matmuls; ++layer) {
                fully_fused_hidden_backward(act_shmem, weights + weights_stride * (hidden_matmuls - layer - 1u), out_intermediate + layer_stride * (layer + 1u) + elem_idx_base * kFusedWidth, forward + layer_stride * (hidden_matmuls - layer - 1u) + elem_idx_base * kFusedWidth);
            }
        }
        __global__ void k_compute_input_grad_prefix(const __half* __restrict__ weights, const __half* __restrict__ backprop, const std::uint32_t rows, const std::uint32_t input_width, const std::uint32_t output_width, const std::uint32_t prefix_width, __half* __restrict__ dinput) {
            __shared__ __half weights_tile[32][16];
            __shared__ __half backprop_tile[16][32];
            const std::uint32_t input_col = blockIdx.x * 16u + threadIdx.x;
            const std::uint32_t row       = blockIdx.y * 16u + threadIdx.y;
            float sum                     = 0.0f;
            const std::uint32_t tid       = threadIdx.y * blockDim.x + threadIdx.x;
            for (std::uint32_t output_base = 0u; output_base < output_width; output_base += 32u) {
                for (std::uint32_t idx = tid; idx < 32u * 16u; idx += blockDim.x * blockDim.y) {
                    const std::uint32_t tile_out    = idx / 16u;
                    const std::uint32_t tile_in     = idx - tile_out * 16u;
                    const std::uint32_t global_out  = output_base + tile_out;
                    const std::uint32_t global_in   = blockIdx.x * 16u + tile_in;
                    weights_tile[tile_out][tile_in] = global_out < output_width && global_in < prefix_width ? weights[static_cast<std::uint64_t>(global_out) * input_width + global_in] : __float2half_rn(0.0f);
                }
                for (std::uint32_t idx = tid; idx < 16u * 32u; idx += blockDim.x * blockDim.y) {
                    const std::uint32_t tile_row      = idx / 32u;
                    const std::uint32_t tile_out      = idx - tile_row * 32u;
                    const std::uint32_t global_row    = blockIdx.y * 16u + tile_row;
                    const std::uint32_t global_out    = output_base + tile_out;
                    backprop_tile[tile_row][tile_out] = global_row < rows && global_out < output_width ? backprop[static_cast<std::uint64_t>(global_row) * output_width + global_out] : __float2half_rn(0.0f);
                }
                __syncthreads();
                if (row < rows && input_col < prefix_width) {
#pragma unroll
                    for (std::uint32_t tile_out = 0u; tile_out < 32u; ++tile_out) {
                        sum += __half2float(backprop_tile[threadIdx.y][tile_out]) * __half2float(weights_tile[tile_out][threadIdx.x]);
                    }
                }
                __syncthreads();
            }
            if (row < rows && input_col < prefix_width) dinput[static_cast<std::uint64_t>(row) * input_width + input_col] = __float2half_rn(sum);
        }
        __device__ __forceinline__ float sigmoid_raw(const float x) {
            return 1.0f / (1.0f + __expf(-x));
        }
        __device__ __forceinline__ float softplus_sigma(const float x) {
            if (x > 20.0f) return x;
            if (x < -20.0f) return __expf(x);
            return log1pf(__expf(x));
        }
        __global__ void k_pack_density_input(const float* __restrict__ encoded_pts, const std::uint32_t rows, const std::uint32_t padded_rows, __half* __restrict__ density_input) {
            const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t total = padded_rows * kDensityInputDim;
            if (idx >= total) return;
            const std::uint32_t row = idx / kDensityInputDim;
            const std::uint32_t col = idx - row * kDensityInputDim;
            float value             = 0.0f;
            if (row < rows && col < kPtsInDim) value = encoded_pts[static_cast<std::uint64_t>(row) * kPtsInDim + col];
            density_input[idx] = __float2half_rn(value);
        }
        __global__ void k_pack_color_input(const __half* __restrict__ density_output, const float* __restrict__ encoded_dir, const std::uint32_t rows, const std::uint32_t padded_rows, __half* __restrict__ color_input) {
            const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t total = padded_rows * kColorInputDim;
            if (idx >= total) return;
            const std::uint32_t row = idx / kColorInputDim;
            const std::uint32_t col = idx - row * kColorInputDim;
            float value             = 0.0f;
            if (row < rows) {
                if (col < kGeoFeatureDim) {
                    value = __half2float(density_output[static_cast<std::uint64_t>(row) * kDensityOutputDim + 1u + col]);
                } else if (col < kGeoFeatureDim + kDirInDim) {
                    value = encoded_dir[static_cast<std::uint64_t>(row) * kDirInDim + (col - kGeoFeatureDim)];
                }
            }
            color_input[idx] = __float2half_rn(value);
        }
        __global__ void k_pack_train_density_input(const nerf::sampler::SampleStep* __restrict__ sample_steps, const nerf::sampler::SampleBatchState* __restrict__ batch_state, const std::uint32_t padded_rows, __half* __restrict__ density_input) {
            const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t total = padded_rows * kDensityInputDim;
            if (idx >= total) return;
            const std::uint32_t row  = idx / kDensityInputDim;
            const std::uint32_t col  = idx - row * kDensityInputDim;
            const std::uint32_t rows = batch_state->sample_step_count;
            if (row >= rows) {
                density_input[idx] = __float2half_rn(0.0f);
                return;
            }
            const nerf::sampler::SampleStep sample = sample_steps[row];
            if (col == 0u) {
                density_input[idx] = __float2half_rn(sample.x);
                return;
            }
            if (col == 1u) {
                density_input[idx] = __float2half_rn(sample.y);
                return;
            }
            if (col == 2u) {
                density_input[idx] = __float2half_rn(sample.z);
                return;
            }
            const std::uint32_t encoded_col = col - 3u;
            const std::uint32_t level       = encoded_col / 6u;
            const std::uint32_t lane        = encoded_col - level * 6u;
            const float freq                = __uint2float_rn(1u << level);
            const float px                  = sample.x * freq;
            const float py                  = sample.y * freq;
            const float pz                  = sample.z * freq;
            float sx                        = 0.0f;
            float cx                        = 0.0f;
            float sy                        = 0.0f;
            float cy                        = 0.0f;
            float sz                        = 0.0f;
            float cz                        = 0.0f;
            __sincosf(px, &sx, &cx);
            __sincosf(py, &sy, &cy);
            __sincosf(pz, &sz, &cz);
            float value = 0.0f;
            switch (lane) {
            case 0u: value = sx; break;
            case 1u: value = sy; break;
            case 2u: value = sz; break;
            case 3u: value = cx; break;
            case 4u: value = cy; break;
            default: value = cz; break;
            }
            density_input[idx] = __float2half_rn(value);
        }
        __global__ void k_pack_train_color_input(const nerf::sampler::SampleStep* __restrict__ sample_steps, const nerf::sampler::SampleBatchState* __restrict__ batch_state, const __half* __restrict__ density_output, const std::uint32_t padded_rows, __half* __restrict__ color_input) {
            const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t total = padded_rows * kColorInputDim;
            if (idx >= total) return;
            const std::uint32_t row  = idx / kColorInputDim;
            const std::uint32_t col  = idx - row * kColorInputDim;
            const std::uint32_t rows = batch_state->sample_step_count;
            if (row >= rows) {
                color_input[idx] = __float2half_rn(0.0f);
                return;
            }
            if (col < kGeoFeatureDim) {
                color_input[idx] = density_output[static_cast<std::uint64_t>(row) * kDensityOutputDim + 1u + col];
                return;
            }
            const nerf::sampler::SampleStep sample = sample_steps[row];
            if (col == kGeoFeatureDim + 0u) {
                color_input[idx] = __float2half_rn(sample.dx);
                return;
            }
            if (col == kGeoFeatureDim + 1u) {
                color_input[idx] = __float2half_rn(sample.dy);
                return;
            }
            if (col == kGeoFeatureDim + 2u) {
                color_input[idx] = __float2half_rn(sample.dz);
                return;
            }
            const std::uint32_t encoded_col = col - (kGeoFeatureDim + 3u);
            const std::uint32_t level       = encoded_col / 6u;
            const std::uint32_t lane        = encoded_col - level * 6u;
            const float freq                = __uint2float_rn(1u << level);
            const float dx                  = sample.dx * freq;
            const float dy                  = sample.dy * freq;
            const float dz                  = sample.dz * freq;
            float sx                        = 0.0f;
            float cx                        = 0.0f;
            float sy                        = 0.0f;
            float cy                        = 0.0f;
            float sz                        = 0.0f;
            float cz                        = 0.0f;
            __sincosf(dx, &sx, &cx);
            __sincosf(dy, &sy, &cy);
            __sincosf(dz, &sz, &cz);
            float value = 0.0f;
            switch (lane) {
            case 0u: value = sx; break;
            case 1u: value = sy; break;
            case 2u: value = sz; break;
            case 3u: value = cx; break;
            case 4u: value = cy; break;
            default: value = cz; break;
            }
            color_input[idx] = __float2half_rn(value);
        }
        __global__ void k_unpack_network_outputs(const __half* __restrict__ density_output, const __half* __restrict__ color_output, const std::uint32_t rows, float* __restrict__ raw_rgb, float* __restrict__ raw_sigma) {
            const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= rows) return;
            raw_sigma[idx]                                         = __half2float(density_output[static_cast<std::uint64_t>(idx) * kDensityOutputDim]);
            raw_rgb[static_cast<std::uint64_t>(idx) * 3ull + 0ull] = __half2float(color_output[static_cast<std::uint64_t>(idx) * kColorOutputPaddedDim + 0ull]);
            raw_rgb[static_cast<std::uint64_t>(idx) * 3ull + 1ull] = __half2float(color_output[static_cast<std::uint64_t>(idx) * kColorOutputPaddedDim + 1ull]);
            raw_rgb[static_cast<std::uint64_t>(idx) * 3ull + 2ull] = __half2float(color_output[static_cast<std::uint64_t>(idx) * kColorOutputPaddedDim + 2ull]);
        }
        __global__ void k_pack_color_output_grad(const float* __restrict__ d_rgb, const nerf::sampler::SampleBatchState* __restrict__ batch_state, const std::uint32_t padded_rows, const float loss_scale, __half* __restrict__ color_doutput) {
            const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t total = padded_rows * kColorOutputPaddedDim;
            if (idx >= total) return;
            const std::uint32_t row  = idx / kColorOutputPaddedDim;
            const std::uint32_t col  = idx - row * kColorOutputPaddedDim;
            const std::uint32_t rows = batch_state->sample_step_count;
            float value              = 0.0f;
            if (row < rows && col < kColorOutputDim) value = d_rgb[static_cast<std::uint64_t>(row) * 3ull + col] * loss_scale;
            color_doutput[idx] = __float2half_rn(value);
        }
        __global__ void k_pack_density_output_grad(const float* __restrict__ d_sigma, const __half* __restrict__ color_dinput, const nerf::sampler::SampleBatchState* __restrict__ batch_state, const std::uint32_t padded_rows, const float loss_scale, __half* __restrict__ density_doutput) {
            const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t total = padded_rows * kDensityOutputDim;
            if (idx >= total) return;
            const std::uint32_t row  = idx / kDensityOutputDim;
            const std::uint32_t col  = idx - row * kDensityOutputDim;
            const std::uint32_t rows = batch_state->sample_step_count;
            float value              = 0.0f;
            if (row < rows) {
                if (col == 0u)
                    value = d_sigma[row] * loss_scale;
                else
                    value = __half2float(color_dinput[static_cast<std::uint64_t>(row) * kColorInputDim + (col - 1u)]);
            }
            density_doutput[idx] = __float2half_rn(value);
        }
        __global__ void k_accumulate_gradients_half(__half* dst, const __half* src, const std::uint32_t n) {
            const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            dst[idx] = __float2half_rn(__half2float(dst[idx]) + __half2float(src[idx]));
        }
        __global__ void k_unpack_density_sigma(const __half* __restrict__ density_output, const std::uint32_t rows, float* __restrict__ raw_sigma) {
            const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= rows) return;
            raw_sigma[idx] = __half2float(density_output[static_cast<std::uint64_t>(idx) * kDensityOutputDim]);
        }
        __global__ void k_ray_march_mse_grad(const float* __restrict__ raw_rgb, const float* __restrict__ raw_sigma, const nerf::sampler::SampleStep* __restrict__ sample_steps, const nerf::sampler::SampleRay* __restrict__ rays, const nerf::sampler::SampleBatchState* __restrict__ batch_state, const std::uint32_t max_ray_count, float* __restrict__ d_raw_rgb, float* __restrict__ d_raw_sigma, float* __restrict__ trans_tmp, float* __restrict__ loss_sum) {
            const std::uint32_t local_ray   = blockIdx.x * blockDim.x + threadIdx.x;
            float ray_loss                  = 0.0f;
            const std::uint32_t active_rays = batch_state->active_ray_count;
            const float inv_norm            = active_rays == 0u ? 0.0f : 1.0f / static_cast<float>(active_rays);
            if (local_ray < active_rays && local_ray < max_ray_count) {
                const nerf::sampler::SampleRay ray = rays[local_ray];
                const std::uint32_t count          = ray.sample_count;
                if (count > 0u) {
                    const std::uint32_t base = ray.sample_begin;
                    float T                  = 1.0f;
                    float accum_r            = 0.0f;
                    float accum_g            = 0.0f;
                    float accum_b            = 0.0f;
                    for (std::uint32_t i = 0; i < count; ++i) {
                        const std::uint32_t s = base + i;
                        const float dt        = fmaxf(0.0f, sample_steps[s].dt);
                        const float sig_raw   = raw_sigma[s];
                        const float sigma     = softplus_sigma(sig_raw);
                        const float a         = 1.0f - __expf(-(sigma * dt));
                        const float r         = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 0ull]);
                        const float g         = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 1ull]);
                        const float b         = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 2ull]);
                        trans_tmp[s]          = T;
                        const float w         = T * a;
                        accum_r += w * r;
                        accum_g += w * g;
                        accum_b += w * b;
                        T *= 1.0f - a;
                    }
                    const float pred_r = accum_r + T;
                    const float pred_g = accum_g + T;
                    const float pred_b = accum_b + T;
                    const float gt_r   = ray.gt_r;
                    const float gt_g   = ray.gt_g;
                    const float gt_b   = ray.gt_b;
                    const float gt_a   = fminf(1.0f, fmaxf(0.0f, ray.gt_a));
                    const float ray_w  = 0.1f + 0.9f * gt_a;
                    const float dr     = pred_r - gt_r;
                    const float dg     = pred_g - gt_g;
                    const float db     = pred_b - gt_b;
                    ray_loss           = ray_w * (dr * dr + dg * dg + db * db);
                    const float g_r    = 2.0f * ray_w * dr * inv_norm;
                    const float g_g    = 2.0f * ray_w * dg * inv_norm;
                    const float g_b    = 2.0f * ray_w * db * inv_norm;
                    float suffix_r     = T;
                    float suffix_g     = T;
                    float suffix_b     = T;
                    for (int i = static_cast<int>(count) - 1; i >= 0; --i) {
                        const std::uint32_t s                                  = base + static_cast<std::uint32_t>(i);
                        const float dt                                         = fmaxf(0.0f, sample_steps[s].dt);
                        const float sig_raw                                    = raw_sigma[s];
                        const float sigma                                      = softplus_sigma(sig_raw);
                        const float exp_neg_sigma_dt                           = __expf(-(sigma * dt));
                        const float a                                          = 1.0f - exp_neg_sigma_dt;
                        const float one_minus_a                                = fmaxf(1e-6f, 1.0f - a);
                        const float r                                          = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 0ull]);
                        const float g                                          = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 1ull]);
                        const float b                                          = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 2ull]);
                        const float trans_i                                    = trans_tmp[s];
                        const float w                                          = trans_i * a;
                        const float dC_da_r                                    = trans_i * r - suffix_r / one_minus_a;
                        const float dC_da_g                                    = trans_i * g - suffix_g / one_minus_a;
                        const float dC_da_b                                    = trans_i * b - suffix_b / one_minus_a;
                        const float dL_da                                      = g_r * dC_da_r + g_g * dC_da_g + g_b * dC_da_b;
                        const float dsigma_draw                                = sigmoid_raw(sig_raw);
                        d_raw_sigma[s]                                         = dL_da * dt * exp_neg_sigma_dt * dsigma_draw;
                        const float wr                                         = w * g_r;
                        const float wg                                         = w * g_g;
                        const float wb                                         = w * g_b;
                        d_raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 0ull] = wr * r * (1.0f - r);
                        d_raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 1ull] = wg * g * (1.0f - g);
                        d_raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 2ull] = wb * b * (1.0f - b);
                        suffix_r += w * r;
                        suffix_g += w * g;
                        suffix_b += w * b;
                    }
                }
            }
            __shared__ float block_loss[256];
            block_loss[threadIdx.x] = ray_loss;
            __syncthreads();
            for (std::uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
                if (threadIdx.x < stride) block_loss[threadIdx.x] += block_loss[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u) atomicAdd(loss_sum, block_loss[0]);
        }
    } // namespace
    bool init_network_module(NetworkSet& network_set, cudaStream_t stream) {
        auto* handle = reinterpret_cast<cublasHandle_t>(network_set.blas_handle);
        if (handle == nullptr) {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
            network_set.blas_handle = handle;
        }
        if (handle == nullptr) return false;
        if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return false;
        if (network_set.blas_workspace != nullptr && network_set.blas_workspace_bytes != 0u)
            if (cublasSetWorkspace(handle, network_set.blas_workspace, network_set.blas_workspace_bytes) != CUBLAS_STATUS_SUCCESS) return false;
        if (cudaFuncSetAttribute(k_fully_fused_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, std::max(kFusedForwardShmemDensity, kFusedForwardShmemColor)) != cudaSuccess) return false;
        if (cudaFuncSetAttribute(k_fully_fused_backward, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedBackwardShmem) != cudaSuccess) return false;
        return true;
    }
    bool describe_network_checkpoint_layout(const NetworkSet& network_set, NetworkCheckpointLayout& layout) {
        layout                             = NetworkCheckpointLayout{};
        layout.density_param_count         = network_set.density.params_f32.count;
        layout.color_param_count           = network_set.color.params_f32.count;
        layout.density_input_width         = network_set.density.input_width;
        layout.density_width               = network_set.density.width;
        layout.density_hidden_layers       = network_set.density.hidden_matmuls + 1u;
        layout.density_output_width        = network_set.density.output_width;
        layout.color_input_width           = network_set.color.input_width;
        layout.color_width                 = network_set.color.width;
        layout.color_hidden_layers         = network_set.color.hidden_matmuls + 1u;
        layout.color_output_width          = network_set.color.output_width;
        std::uint32_t tensor_index         = 0u;
        std::uint64_t density_offset       = 0u;
        std::uint64_t color_offset         = 0u;
        const std::uint32_t density_layers = network_set.density.hidden_matmuls + 2u;
        const std::uint32_t color_layers   = network_set.color.hidden_matmuls + 2u;
        if (density_layers + color_layers > 32u) return false;
        for (std::uint32_t layer_index = 0u; layer_index < density_layers; ++layer_index) {
            NetworkCheckpointTensorLayout& tensor = layout.tensors[tensor_index++];
            tensor.network_index                  = 0u;
            tensor.offset                         = density_offset;
            tensor.cols                           = layer_index == 0u ? network_set.density.input_width : network_set.density.width;
            tensor.rows                           = layer_index == density_layers - 1u ? network_set.density.output_width : network_set.density.width;
            if (layer_index == 0u)
                std::snprintf(tensor.name, sizeof(tensor.name), "density.input.weight");
            else if (layer_index == density_layers - 1u)
                std::snprintf(tensor.name, sizeof(tensor.name), "density.output.weight");
            else
                std::snprintf(tensor.name, sizeof(tensor.name), "density.hidden.%u.weight", layer_index - 1u);
            density_offset += static_cast<std::uint64_t>(tensor.rows) * tensor.cols;
        }
        for (std::uint32_t layer_index = 0u; layer_index < color_layers; ++layer_index) {
            NetworkCheckpointTensorLayout& tensor = layout.tensors[tensor_index++];
            tensor.network_index                  = 1u;
            tensor.offset                         = color_offset;
            tensor.cols                           = layer_index == 0u ? network_set.color.input_width : network_set.color.width;
            tensor.rows                           = layer_index == color_layers - 1u ? network_set.color.output_width : network_set.color.width;
            if (layer_index == 0u)
                std::snprintf(tensor.name, sizeof(tensor.name), "color.input.weight");
            else if (layer_index == color_layers - 1u)
                std::snprintf(tensor.name, sizeof(tensor.name), "color.output.weight");
            else
                std::snprintf(tensor.name, sizeof(tensor.name), "color.hidden.%u.weight", layer_index - 1u);
            color_offset += static_cast<std::uint64_t>(tensor.rows) * tensor.cols;
        }
        if (density_offset != network_set.density.params_f32.count) return false;
        if (color_offset != network_set.color.params_f32.count) return false;
        layout.tensor_count = tensor_index;
        return true;
    }
    static bool fully_fused_mlp_inference(const FusedNetworkState& network, cudaStream_t stream, const __half* input, std::uint32_t rows, __half* output) {
        if (network.params.ptr == nullptr || input == nullptr || output == nullptr) return false;
        if (rows == 0u || rows % kFusedBatchQuantum != 0u) return false;
        constexpr dim3 threads           = {32u, 8u, 1u};
        const dim3 blocks            = {rows / kFusedBatchQuantum, 1u, 1u};
        const std::size_t shmem_size = std::max(sizeof(__half) * (kFusedWidth + 16u) * (network.input_width + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));
        k_fully_fused_forward<<<blocks, threads, shmem_size, stream>>>(input, network.params.ptr, nullptr, output, network.input_width, network.hidden_matmuls);
        return cudaGetLastError() == cudaSuccess;
    }
    static bool fully_fused_mlp_forward(const FusedNetworkState& network, cudaStream_t stream, const __half* input, std::uint32_t rows, __half* output, __half* forward_hidden) {
        if (network.params.ptr == nullptr || input == nullptr || output == nullptr || forward_hidden == nullptr) return false;
        if (rows == 0u || rows % kFusedBatchQuantum != 0u) return false;
        constexpr dim3 threads           = {32u, 8u, 1u};
        const dim3 blocks            = {rows / kFusedBatchQuantum, 1u, 1u};
        const std::size_t shmem_size = std::max(sizeof(__half) * (kFusedWidth + 16u) * (network.input_width + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));
        k_fully_fused_forward<<<blocks, threads, shmem_size, stream>>>(input, network.params.ptr, forward_hidden, output, network.input_width, network.hidden_matmuls);
        return cudaGetLastError() == cudaSuccess;
    }
    static bool fully_fused_mlp_backward(const FusedNetworkState& network, void* blas_handle, cudaStream_t stream, const __half* input, const __half* doutput, std::uint32_t rows, const std::uint32_t dinput_prefix_width, __half* dinput, __half* backward_hidden, const __half* forward_hidden) {
        auto* handle = reinterpret_cast<cublasHandle_t>(blas_handle);
        if (handle == nullptr || network.params.ptr == nullptr || network.gradients_tmp.ptr == nullptr || input == nullptr || doutput == nullptr || forward_hidden == nullptr || backward_hidden == nullptr) return false;
        if (rows == 0u || rows % kFusedBatchQuantum != 0u) return false;
        constexpr dim3 threads           = {32u, 8u, 1u};
        const dim3 blocks            = {rows / kFusedBatchQuantum, 1u, 1u};
        constexpr std::size_t shmem_size = kFusedBackwardShmem;
        k_fully_fused_backward<<<blocks, threads, shmem_size, stream>>>(doutput, network.params.ptr + static_cast<std::uint64_t>(kFusedWidth) * network.input_width, backward_hidden, forward_hidden, rows, network.hidden_matmuls);
        if (cudaGetLastError() != cudaSuccess) return false;
        constexpr float alpha                  = 1.0f;
        constexpr float beta                   = 0.0f;
        std::uint64_t gradient_offset      = 0u;
        const __half* const first_hidden   = forward_hidden;
        const __half* const last_hidden    = forward_hidden + static_cast<std::uint64_t>(network.hidden_matmuls) * kFusedWidth * rows;
        const __half* const first_backprop = backward_hidden;
        const __half* const last_backprop  = backward_hidden + static_cast<std::uint64_t>(network.hidden_matmuls) * kFusedWidth * rows;
        constexpr dim3 input_grad_threads{16u, 16u, 1u};
        gradient_offset += static_cast<std::uint64_t>(kFusedWidth) * network.input_width;
        gradient_offset += static_cast<std::uint64_t>(network.hidden_matmuls) * kFusedWidth * kFusedWidth;
        if (cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, kFusedWidth, kFusedOutputWidth, static_cast<int>(rows), &alpha, last_hidden, CUDA_R_16F, kFusedWidth, doutput, CUDA_R_16F, kFusedOutputWidth, &beta, network.gradients_tmp.ptr + gradient_offset, CUDA_R_16F, kFusedWidth, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
        gradient_offset = static_cast<std::uint64_t>(kFusedWidth) * network.input_width;
        for (std::uint32_t layer = 0u; layer < network.hidden_matmuls; ++layer) {
            const std::uint32_t weight_index = network.hidden_matmuls - 1u - layer;
            const __half* const activations  = first_hidden + static_cast<std::uint64_t>(weight_index) * kFusedWidth * rows;
            const __half* const backprop     = first_backprop + static_cast<std::uint64_t>(layer) * kFusedWidth * rows;
            const std::uint64_t layer_offset = gradient_offset + static_cast<std::uint64_t>(weight_index) * kFusedWidth * kFusedWidth;
            if (cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, kFusedWidth, kFusedWidth, static_cast<int>(rows), &alpha, activations, CUDA_R_16F, kFusedWidth, backprop, CUDA_R_16F, kFusedWidth, &beta, network.gradients_tmp.ptr + layer_offset, CUDA_R_16F, kFusedWidth, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
        }
        if (cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(network.input_width), kFusedWidth, static_cast<int>(rows), &alpha, input, CUDA_R_16F, static_cast<int>(network.input_width), last_backprop, CUDA_R_16F, kFusedWidth, &beta, network.gradients_tmp.ptr, CUDA_R_16F, static_cast<int>(network.input_width), CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
        if (dinput != nullptr && dinput_prefix_width != 0u) {
            k_compute_input_grad_prefix<<<dim3{(dinput_prefix_width + 15u) / 16u, (rows + 15u) / 16u, 1u}, input_grad_threads, 0, stream>>>(network.params.ptr, last_backprop, rows, network.input_width, kFusedWidth, dinput_prefix_width, dinput);
            if (cudaGetLastError() != cudaSuccess) return false;
        }
        return true;
    }
    bool run_network_inference(NetworkSet& network_set, NetworkWorkspace& workspace, cudaStream_t stream, const NetworkInferenceRequest& request) {
        const std::uint32_t rows          = request.rows;
        const std::uint32_t padded_rows   = (rows + kNetworkBatchGranularity - 1u) / kNetworkBatchGranularity * kNetworkBatchGranularity;
        constexpr std::uint32_t threads       = 256u;
        const std::uint32_t density_total = padded_rows * kDensityInputDim;
        const std::uint32_t color_total   = padded_rows * kColorInputDim;
        const std::uint32_t output_blocks = (rows + threads - 1u) / threads;
        k_pack_density_input<<<(density_total + threads - 1u) / threads, threads, 0, stream>>>(request.encoded_pts, rows, padded_rows, workspace.density_input);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (!fully_fused_mlp_inference(network_set.density, stream, workspace.density_input, padded_rows, workspace.density_output)) return false;
        k_pack_color_input<<<(color_total + threads - 1u) / threads, threads, 0, stream>>>(workspace.density_output, request.encoded_dir, rows, padded_rows, workspace.color_input);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (!fully_fused_mlp_inference(network_set.color, stream, workspace.color_input, padded_rows, workspace.color_output)) return false;
        k_unpack_network_outputs<<<output_blocks, threads, 0, stream>>>(workspace.density_output, workspace.color_output, rows, request.raw_rgb, request.raw_sigma);
        return cudaGetLastError() == cudaSuccess;
    }
    bool run_density_inference(NetworkSet& network_set, NetworkWorkspace& workspace, cudaStream_t stream, const float* encoded_pts, const std::uint32_t rows, float* raw_sigma) {
        const std::uint32_t padded_rows   = (rows + kNetworkBatchGranularity - 1u) / kNetworkBatchGranularity * kNetworkBatchGranularity;
        constexpr std::uint32_t threads       = 256u;
        const std::uint32_t density_total = padded_rows * kDensityInputDim;
        k_pack_density_input<<<(density_total + threads - 1u) / threads, threads, 0, stream>>>(encoded_pts, rows, padded_rows, workspace.density_input);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (!fully_fused_mlp_inference(network_set.density, stream, workspace.density_input, padded_rows, workspace.density_output)) return false;
        k_unpack_density_sigma<<<(rows + threads - 1u) / threads, threads, 0, stream>>>(workspace.density_output, rows, raw_sigma);
        return cudaGetLastError() == cudaSuccess;
    }
    bool run_network_training(NetworkSet& network_set, NetworkWorkspace& workspace, cudaStream_t stream, const NetworkTrainingRequest& request, const float loss_scale) {
        const std::uint32_t padded_rows = (request.max_sample_count + kNetworkBatchGranularity - 1u) / kNetworkBatchGranularity * kNetworkBatchGranularity;
        constexpr std::uint32_t threads     = 256u;
        bool ok                         = false;
        do {
            k_pack_train_density_input<<<(padded_rows * kDensityInputDim + threads - 1u) / threads, threads, 0, stream>>>(request.sample_steps, request.batch_state, padded_rows, workspace.density_input);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_forward(network_set.density, stream, workspace.density_input, padded_rows, workspace.density_output, workspace.density_forward_hidden)) break;
            k_pack_train_color_input<<<(padded_rows * kColorInputDim + threads - 1u) / threads, threads, 0, stream>>>(request.sample_steps, request.batch_state, workspace.density_output, padded_rows, workspace.color_input);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_forward(network_set.color, stream, workspace.color_input, padded_rows, workspace.color_output, workspace.color_forward_hidden)) break;
            k_unpack_network_outputs<<<(request.max_sample_count + threads - 1u) / threads, threads, 0, stream>>>(workspace.density_output, workspace.color_output, request.max_sample_count, workspace.raw_rgb, workspace.raw_sigma);
            if (cudaGetLastError() != cudaSuccess) break;
            if (cudaMemsetAsync(workspace.d_rgb, 0, static_cast<std::uint64_t>(request.max_sample_count) * 3ull * sizeof(float), stream) != cudaSuccess) break;
            if (cudaMemsetAsync(workspace.d_sigma, 0, static_cast<std::uint64_t>(request.max_sample_count) * sizeof(float), stream) != cudaSuccess) break;
            k_ray_march_mse_grad<<<(request.max_ray_count + threads - 1u) / threads, threads, 0, stream>>>(workspace.raw_rgb, workspace.raw_sigma, request.sample_steps, request.sample_rays, request.batch_state, request.max_ray_count, workspace.d_rgb, workspace.d_sigma, workspace.trans_tmp, workspace.loss_sum);
            if (cudaGetLastError() != cudaSuccess) break;
            k_pack_color_output_grad<<<(padded_rows * kColorOutputPaddedDim + threads - 1u) / threads, threads, 0, stream>>>(workspace.d_rgb, request.batch_state, padded_rows, loss_scale, workspace.color_doutput);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_backward(network_set.color, network_set.blas_handle, stream, workspace.color_input, workspace.color_doutput, padded_rows, kGeoFeatureDim, workspace.color_dinput, workspace.color_backward_hidden, workspace.color_forward_hidden)) break;
            k_accumulate_gradients_half<<<(static_cast<std::uint32_t>(network_set.color.gradients.count) + threads - 1u) / threads, threads, 0, stream>>>(network_set.color.gradients.ptr, network_set.color.gradients_tmp.ptr, static_cast<std::uint32_t>(network_set.color.gradients.count));
            k_pack_density_output_grad<<<(padded_rows * kDensityOutputDim + threads - 1u) / threads, threads, 0, stream>>>(workspace.d_sigma, workspace.color_dinput, request.batch_state, padded_rows, loss_scale, workspace.density_doutput);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_backward(network_set.density, network_set.blas_handle, stream, workspace.density_input, workspace.density_doutput, padded_rows, 0u, nullptr, workspace.density_backward_hidden, workspace.density_forward_hidden)) break;
            k_accumulate_gradients_half<<<(static_cast<std::uint32_t>(network_set.density.gradients.count) + threads - 1u) / threads, threads, 0, stream>>>(network_set.density.gradients.ptr, network_set.density.gradients_tmp.ptr, static_cast<std::uint32_t>(network_set.density.gradients.count));
            ok = cudaGetLastError() == cudaSuccess;
        } while (false);
        return ok;
    }
} // namespace nerf::network


namespace nerf::runtime {
    using namespace nerf::compile_time;

    struct TrainStepConfig {
        float learning_rate;
        float adam_beta1;
        float adam_beta2;
        float adam_eps;
        float grad_clip_norm;
        float update_guard_grad_norm;
        float loss_scale;
        std::uint32_t lr_decay_ksteps;
    };
    struct OccupancyUpdateRequest {
        std::uint32_t* bitfield          = nullptr;
        std::uint64_t bitfield_bytes     = 0u;
        float* density_grid              = nullptr;
        std::uint32_t grid_res           = 0u;
        std::uint32_t cell_count         = 0u;
        std::uint32_t update_count       = 0u;
        std::uint32_t update_rows_padded = 0u;
        float decay                      = 0.98f;
        float threshold                  = 0.01f;
        std::uint32_t update_interval    = 1u;
        std::uint32_t warmup_steps       = 32u;
        float3 aabb_min{};
        float3 aabb_max{};
    };
    struct TrainingGraphConfig {
        const nerf::sampler::SampleRay* sample_rays        = nullptr;
        const nerf::sampler::SampleStep* sample_steps      = nullptr;
        const nerf::sampler::SampleBatchState* batch_state = nullptr;
        std::uint32_t image_count                          = 0u;
        std::uint32_t max_ray_count                        = 0u;
        std::uint32_t max_sample_count                     = 0u;
        TrainStepConfig train_cfg{};
    };
    struct TrainRuntime {
        cudaStream_t stream                   = nullptr;
        cudaGraph_t train_step_graph          = nullptr;
        cudaGraphExec_t train_step_graph_exec = nullptr;
        nerf::network::NetworkSet network{};
        nerf::network::NetworkWorkspace workspace{};
        TrainingDeviceState* device_state = nullptr;
        std::uint32_t host_generation     = 0u;
        bool training_configured          = false;
        NerfTrainingConfig training_config{};
        std::mutex run_mutex{};
    };
    struct DeviceContext {
        std::mutex train_runtime_mutex{};
        std::shared_ptr<TrainRuntime> train_runtime{};
    };
    struct Region {
        std::uint64_t offset_bytes = 0u;
        std::uint64_t size_bytes   = 0u;
    };
    struct DeviceSpan {
        std::byte* ptr           = nullptr;
        std::uint64_t size_bytes = 0u;
    };
    struct ContextStorage {
        DeviceContext* cuda_context    = nullptr;
        std::byte* scratch_device_base = nullptr;
        std::byte* scene_device_base   = nullptr;
        DeviceSpan images{};
        DeviceSpan xforms{};
        DeviceSpan inference_rgba8{};
        DeviceSpan occupancy_bitfield{};
        DeviceSpan occupancy_density{};
        DeviceSpan sample_rays{};
        DeviceSpan sample_steps{};
        DeviceSpan sample_batch_state{};
        nerf::host::DatasetInfo dataset_info{};
        std::uint32_t occupancy_grid_res    = 0u;
        std::uint32_t max_sample_steps      = 0u;
        std::uint32_t max_batch_rays        = 0u;
        std::uint64_t arena_alignment_bytes = 0u;
    };
    __host__ __device__ __forceinline__ std::uint32_t hash_u32(std::uint32_t x) {
        x ^= x >> 16u;
        x *= 0x7feb352du;
        x ^= x >> 15u;
        x *= 0x846ca68bu;
        x ^= x >> 16u;
        return x;
    }
    __global__ void k_float_to_half(const float* src, __half* dst, std::uint32_t n) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) dst[idx] = __float2half_rn(src[idx]);
    }
    __global__ void k_accum_grad_stats_half(const __half* grads, const std::uint32_t n, float* grad_sumsq, std::uint32_t* nonfinite_flag) {
        __shared__ float block_sum[256];
        __shared__ std::uint32_t block_has_nonfinite;
        const std::uint32_t tid    = threadIdx.x;
        const std::uint32_t idx    = blockIdx.x * blockDim.x + tid;
        const std::uint32_t stride = blockDim.x * gridDim.x;
        if (tid == 0u) block_has_nonfinite = 0u;
        __syncthreads();
        float local_sum = 0.0f;
        for (std::uint32_t i = idx; i < n; i += stride) {
            const float v = __half2float(grads[i]);
            if (!isfinite(v)) {
                atomicOr(&block_has_nonfinite, 1u);
                continue;
            }
            local_sum += v * v;
        }
        block_sum[tid] = local_sum;
        __syncthreads();
        for (std::uint32_t offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
            if (tid < offset) block_sum[tid] += block_sum[tid + offset];
            __syncthreads();
        }
        if (tid == 0u) {
            if (block_sum[0] != 0.0f) atomicAdd(grad_sumsq, block_sum[0]);
            if (block_has_nonfinite != 0u) atomicExch(nonfinite_flag, 1u);
        }
    }
    __global__ void k_begin_training_step(TrainingDeviceState* state, const std::uint32_t camera_count, const std::uint32_t warmup_steps, const std::uint32_t update_interval) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) return;
        const std::uint32_t frame_index = state->frame_index;
        state->train_camera_idx         = nerf::runtime::hash_u32(frame_index * 747796405u + 2891336453u) % camera_count;
        if (frame_index < warmup_steps) {
            state->occupancy_mode = frame_index == 0u ? kOccupancyModeWarmupInit : kOccupancyModeNone;
            return;
        }
        state->occupancy_mode = (update_interval != 0u && frame_index % update_interval == 0u) ? kOccupancyModeRefresh : kOccupancyModeNone;
    }
    __global__ void k_select_training_bucket(cudaGraphConditionalHandle handle, const nerf::sampler::SampleBatchState* batch_state, const std::uint32_t bucket_rows, const std::uint32_t bucket_count) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) return;
        const std::uint32_t sample_count = batch_state->sample_step_count;
        const std::uint32_t bucket       = sample_count == 0u ? 0u : min(bucket_count, (sample_count + bucket_rows - 1u) / bucket_rows);
        cudaGraphSetConditional(handle, bucket);
    }
    __global__ void k_finalize_training_stats(TrainingDeviceState* state, const nerf::sampler::SampleBatchState* batch_state, const float* loss_sum, const float* grad_sumsq, const std::uint32_t* nonfinite_flag, const float loss_scale) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) return;
        const std::uint32_t active_rays   = batch_state->active_ray_count;
        const float inv_norm              = active_rays == 0u ? 0.0f : 1.0f / static_cast<float>(active_rays);
        const float loss                  = *loss_sum * inv_norm;
        const float grad_norm             = sqrtf(fmaxf(0.0f, *grad_sumsq)) / loss_scale;
        const std::uint32_t has_nonfinite = *nonfinite_flag != 0u || !isfinite(loss) || !isfinite(grad_norm) ? 1u : 0u;
        state->optimizer_step             = state->frame_index + 1u;
        state->stats.loss                 = loss;
        state->stats.grad_norm            = grad_norm;
        state->stats.has_nonfinite        = has_nonfinite;
        state->stats.completed_steps      = state->frame_index + 1u;
        state->stats.last_train_camera    = state->train_camera_idx;
    }
    __global__ void k_commit_training_step(TrainingDeviceState* state) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) return;
        ++state->frame_index;
    }
    __global__ void k_prepare_adam_step_scalars(const TrainingDeviceState* state, const float base_learning_rate, const float beta1, const float beta2, const std::uint32_t lr_decay_ksteps, const float grad_clip_norm, const float update_guard_grad_norm, const float loss_scale, nerf::network::AdamStepScalars* out) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) return;
        const float grad_norm              = state->stats.grad_norm;
        const std::uint32_t has_nonfinite  = state->stats.has_nonfinite;
        const std::uint32_t optimizer_step = state->optimizer_step;
        const bool skip_update             = has_nonfinite != 0u || !isfinite(grad_norm) || grad_norm > update_guard_grad_norm;
        float learning_rate                = 0.0f;
        float grad_scale                   = 1.0f;
        float inv_bias_correction1         = 1.0f;
        float inv_bias_correction2         = 1.0f;
        if (!skip_update) {
            const float decay    = static_cast<float>(lr_decay_ksteps) * 1000.0f;
            learning_rate        = decay > 0.0f ? base_learning_rate * powf(0.1f, static_cast<float>(optimizer_step) / decay) : base_learning_rate;
            grad_scale           = grad_norm > grad_clip_norm && grad_norm > 0.0f ? grad_clip_norm / grad_norm : 1.0f;
            inv_bias_correction1 = 1.0f / (1.0f - powf(beta1, static_cast<float>(optimizer_step)));
            inv_bias_correction2 = 1.0f / (1.0f - powf(beta2, static_cast<float>(optimizer_step)));
        }
        out->learning_rate        = learning_rate;
        out->grad_scale           = grad_scale;
        out->inv_bias_correction1 = inv_bias_correction1;
        out->inv_bias_correction2 = inv_bias_correction2;
        out->inv_loss_scale       = 1.0f / loss_scale;
        out->skip_update          = skip_update ? 1u : 0u;
    }
    __global__ void k_adam_step_half(float* params_f32, __half* params, const __half* grads, float* adam_m, float* adam_v, const std::uint32_t n, const float beta1, const float beta2, const float epsilon, const nerf::network::AdamStepScalars* step_scalars) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        __shared__ nerf::network::AdamStepScalars shared_step;
        if (threadIdx.x == 0u) shared_step = *step_scalars;
        __syncthreads();
        if (shared_step.skip_update != 0u) return;
        const float grad        = __half2float(grads[idx]) * shared_step.inv_loss_scale;
        const float scaled_grad = grad * shared_step.grad_scale;
        const float m           = beta1 * adam_m[idx] + (1.0f - beta1) * scaled_grad;
        const float v           = beta2 * adam_v[idx] + (1.0f - beta2) * scaled_grad * scaled_grad;
        const float step        = shared_step.learning_rate * (m * shared_step.inv_bias_correction1) / (sqrtf(v * shared_step.inv_bias_correction2) + epsilon);
        const float w           = params_f32[idx] - step;
        adam_m[idx]             = m;
        adam_v[idx]             = v;
        params_f32[idx]         = w;
        params[idx]             = __float2half_rn(w);
    }
    __global__ void k_update_occupancy_density(float* data, const std::uint32_t count, const float decay, const TrainingDeviceState* state) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;
        const std::uint32_t occupancy_mode = state->occupancy_mode;
        if (occupancy_mode == kOccupancyModeWarmupInit) {
            data[idx] = 1.0f;
            return;
        }
        if (occupancy_mode == kOccupancyModeRefresh) data[idx] *= decay;
    }
    __global__ void k_update_occupancy_bitfield(std::uint32_t* data, const std::uint32_t count, const TrainingDeviceState* state) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;
        const std::uint32_t occupancy_mode = state->occupancy_mode;
        if (occupancy_mode == kOccupancyModeWarmupInit) {
            data[idx] = 0xFFFFFFFFu;
            return;
        }
        if (occupancy_mode == kOccupancyModeRefresh) data[idx] = 0u;
    }
    __global__ void k_build_occ_inputs(const TrainingDeviceState* state, const std::uint32_t rows, const std::uint32_t update_count, const std::uint32_t cell_count, const std::uint32_t grid_res, const float3 aabb_min, const float3 aabb_max, float* out_inputs) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= rows) return;
        float* dst = out_inputs + static_cast<std::uint64_t>(idx) * 7ull;
        if (state->occupancy_mode != kOccupancyModeRefresh) {
            dst[0] = 0.0f;
            dst[1] = 0.0f;
            dst[2] = 0.0f;
            dst[3] = 0.0f;
            dst[4] = 0.0f;
            dst[5] = 0.0f;
            dst[6] = 1.0f;
            return;
        }
        if (idx >= update_count) {
            dst[0] = 0.0f;
            dst[1] = 0.0f;
            dst[2] = 0.0f;
            dst[3] = 0.0f;
            dst[4] = 0.0f;
            dst[5] = 0.0f;
            dst[6] = 1.0f;
            return;
        }
        const std::uint32_t frame_index     = state->frame_index;
        const std::uint32_t start_cell_base = static_cast<std::uint32_t>(static_cast<std::uint64_t>(frame_index) * update_count % cell_count);
        const std::uint32_t cell            = (start_cell_base + idx) % cell_count;
        const std::uint32_t layer           = grid_res * grid_res;
        const std::uint32_t z               = cell / layer;
        const std::uint32_t rem             = cell - z * layer;
        const std::uint32_t y               = rem / grid_res;
        const std::uint32_t x               = rem - y * grid_res;
        const float3 step                   = make_float3((aabb_max.x - aabb_min.x) / static_cast<float>(grid_res), (aabb_max.y - aabb_min.y) / static_cast<float>(grid_res), (aabb_max.z - aabb_min.z) / static_cast<float>(grid_res));
        const float3 p                      = make_float3(aabb_min.x + (static_cast<float>(x) + 0.5f) * step.x, aabb_min.y + (static_cast<float>(y) + 0.5f) * step.y, aabb_min.z + (static_cast<float>(z) + 0.5f) * step.z);
        dst[0]                              = p.x;
        dst[1]                              = p.y;
        dst[2]                              = p.z;
        dst[3]                              = 0.0f;
        dst[4]                              = 0.0f;
        dst[5]                              = 0.0f;
        dst[6]                              = 1.0f;
    }
    __global__ void k_update_density_from_sigma(float* density_grid, const float* raw_sigma, const TrainingDeviceState* state, const std::uint32_t update_count, const std::uint32_t cell_count) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= update_count) return;
        if (state->occupancy_mode != kOccupancyModeRefresh) return;
        const std::uint32_t frame_index     = state->frame_index;
        const std::uint32_t start_cell_base = static_cast<std::uint32_t>(static_cast<std::uint64_t>(frame_index) * update_count % cell_count);
        const std::uint32_t cell            = (start_cell_base + idx) % cell_count;
        const float sig_raw                 = raw_sigma[idx];
        const float sigma                   = sig_raw > 20.0f ? sig_raw : sig_raw < -20.0f ? __expf(sig_raw) : log1pf(__expf(sig_raw));
        density_grid[cell]                  = fmaxf(density_grid[cell], sigma);
    }
    __global__ void k_rebuild_occ_from_density(const float* density_grid, const TrainingDeviceState* state, const std::uint32_t cell_count, const float threshold, std::uint32_t* bitfield) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= cell_count) return;
        if (state->occupancy_mode != kOccupancyModeRefresh) return;
        if (density_grid[idx] <= threshold) return;
        atomicOr(&bitfield[idx >> 5u], 1u << (idx & 31u));
    }
    static void free_workspace(nerf::network::NetworkWorkspace& workspace) {
        if (workspace.arena) {
            (void) cudaFree(workspace.arena);
            workspace.arena = nullptr;
        }
        workspace = nerf::network::NetworkWorkspace{};
    }
    static void destroy_runtime(TrainRuntime& rt) {
        free_workspace(rt.workspace);
        if (rt.train_step_graph_exec) {
            (void) cudaGraphExecDestroy(rt.train_step_graph_exec);
            rt.train_step_graph_exec = nullptr;
        }
        if (rt.train_step_graph) {
            (void) cudaGraphDestroy(rt.train_step_graph);
            rt.train_step_graph = nullptr;
        }
        if (rt.network.blas_handle) {
            (void) cublasDestroy(reinterpret_cast<cublasHandle_t>(rt.network.blas_handle));
            rt.network.blas_handle = nullptr;
        }
        if (rt.network.blas_workspace) {
            (void) cudaFree(rt.network.blas_workspace);
            rt.network.blas_workspace       = nullptr;
            rt.network.blas_workspace_bytes = 0u;
        }
        if (rt.stream) {
            (void) cudaStreamDestroy(rt.stream);
            rt.stream = nullptr;
        }
        if (rt.device_state) {
            (void) cudaFree(rt.device_state);
            rt.device_state = nullptr;
        }
        if (rt.network.density.params_f32.ptr) (void) cudaFree(rt.network.density.params_f32.ptr);
        if (rt.network.density.params.ptr) (void) cudaFree(rt.network.density.params.ptr);
        if (rt.network.density.gradients.ptr) (void) cudaFree(rt.network.density.gradients.ptr);
        if (rt.network.density.gradients_tmp.ptr) (void) cudaFree(rt.network.density.gradients_tmp.ptr);
        if (rt.network.density.adam_m.ptr) (void) cudaFree(rt.network.density.adam_m.ptr);
        if (rt.network.density.adam_v.ptr) (void) cudaFree(rt.network.density.adam_v.ptr);
        if (rt.network.color.params_f32.ptr) (void) cudaFree(rt.network.color.params_f32.ptr);
        if (rt.network.color.params.ptr) (void) cudaFree(rt.network.color.params.ptr);
        if (rt.network.color.gradients.ptr) (void) cudaFree(rt.network.color.gradients.ptr);
        if (rt.network.color.gradients_tmp.ptr) (void) cudaFree(rt.network.color.gradients_tmp.ptr);
        if (rt.network.color.adam_m.ptr) (void) cudaFree(rt.network.color.adam_m.ptr);
        if (rt.network.color.adam_v.ptr) (void) cudaFree(rt.network.color.adam_v.ptr);
        rt.network             = nerf::network::NetworkSet{};
        rt.host_generation     = 0u;
        rt.training_configured = false;
        rt.training_config     = NerfTrainingConfig{};
    }
    static bool alloc_workspace(nerf::network::NetworkWorkspace& workspace, const std::uint32_t rows) {
        const std::uint32_t padded_rows = (rows + kNetworkBatchGranularity - 1u) / kNetworkBatchGranularity * kNetworkBatchGranularity;
        if (workspace.rows_capacity >= padded_rows) return true;
        free_workspace(workspace);
        const std::uint64_t row_count = padded_rows;
        const auto align_up           = [](const std::uint64_t value, const std::uint64_t alignment) -> std::uint64_t { return value + alignment - 1u & ~(alignment - 1u); };
        std::uint64_t total           = 0u;
        const auto reserve            = [&](const std::uint64_t bytes) {
            total = align_up(total, 16u);
            total += bytes;
        };
        reserve(row_count * 7u * sizeof(float));
        reserve(row_count * kPtsInDim * sizeof(float));
        reserve(row_count * kDirInDim * sizeof(float));
        reserve(row_count * 3u * sizeof(float));
        reserve(row_count * sizeof(float));
        reserve(row_count * 3u * sizeof(float));
        reserve(row_count * sizeof(float));
        reserve(row_count * sizeof(float));
        reserve(sizeof(float));
        reserve(sizeof(float));
        reserve(sizeof(std::uint32_t));
        reserve(row_count * sizeof(std::uint32_t));
        reserve(sizeof(nerf::network::AdamStepScalars));
        reserve(row_count * kDensityInputDim * sizeof(__half));
        reserve(row_count * kDensityOutputDim * sizeof(__half));
        reserve(row_count * kDensityOutputDim * sizeof(__half));
        reserve(static_cast<std::uint64_t>(kDensityHiddenLayers) * row_count * kDensityWidth * sizeof(__half));
        reserve(static_cast<std::uint64_t>(kDensityHiddenLayers) * row_count * kDensityWidth * sizeof(__half));
        reserve(row_count * kColorInputDim * sizeof(__half));
        reserve(row_count * kColorOutputPaddedDim * sizeof(__half));
        reserve(row_count * kColorOutputPaddedDim * sizeof(__half));
        reserve(row_count * kColorInputDim * sizeof(__half));
        reserve(static_cast<std::uint64_t>(kColorHiddenLayers) * row_count * kColorWidth * sizeof(__half));
        reserve(static_cast<std::uint64_t>(kColorHiddenLayers) * row_count * kColorWidth * sizeof(__half));
        if (cudaMalloc(&workspace.arena, total) != cudaSuccess) return false;
        std::uint64_t offset = 0u;
        const auto place     = [&]<typename T0>(T0& field, const std::uint64_t bytes) {
            offset = align_up(offset, 16u);
            field  = reinterpret_cast<std::remove_reference_t<T0>>(workspace.arena + offset);
            offset += bytes;
        };
        place(workspace.inputs_tmp, row_count * 7u * sizeof(float));
        place(workspace.enc_pts, row_count * kPtsInDim * sizeof(float));
        place(workspace.enc_dir, row_count * kDirInDim * sizeof(float));
        place(workspace.raw_rgb, row_count * 3u * sizeof(float));
        place(workspace.raw_sigma, row_count * sizeof(float));
        place(workspace.d_rgb, row_count * 3u * sizeof(float));
        place(workspace.d_sigma, row_count * sizeof(float));
        place(workspace.trans_tmp, row_count * sizeof(float));
        place(workspace.loss_sum, sizeof(float));
        place(workspace.grad_sumsq, sizeof(float));
        place(workspace.nonfinite_flag, sizeof(std::uint32_t));
        place(workspace.ray_counts_tmp, row_count * sizeof(std::uint32_t));
        place(workspace.adam_step_scalars, sizeof(nerf::network::AdamStepScalars));
        place(workspace.density_input, row_count * kDensityInputDim * sizeof(__half));
        place(workspace.density_output, row_count * kDensityOutputDim * sizeof(__half));
        place(workspace.density_doutput, row_count * kDensityOutputDim * sizeof(__half));
        place(workspace.density_forward_hidden, static_cast<std::uint64_t>(kDensityHiddenLayers) * row_count * kDensityWidth * sizeof(__half));
        place(workspace.density_backward_hidden, static_cast<std::uint64_t>(kDensityHiddenLayers) * row_count * kDensityWidth * sizeof(__half));
        place(workspace.color_input, row_count * kColorInputDim * sizeof(__half));
        place(workspace.color_output, row_count * kColorOutputPaddedDim * sizeof(__half));
        place(workspace.color_doutput, row_count * kColorOutputPaddedDim * sizeof(__half));
        place(workspace.color_dinput, row_count * kColorInputDim * sizeof(__half));
        place(workspace.color_forward_hidden, static_cast<std::uint64_t>(kColorHiddenLayers) * row_count * kColorWidth * sizeof(__half));
        place(workspace.color_backward_hidden, static_cast<std::uint64_t>(kColorHiddenLayers) * row_count * kColorWidth * sizeof(__half));
        workspace.rows_capacity = padded_rows;
        return true;
    }
    static bool init_runtime(TrainRuntime& rt) {
        try {
            int device = 0;
            cudaDeviceProp prop{};
            if (cudaGetDevice(&device) != cudaSuccess) return false;
            if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
            if (prop.major * 10 + prop.minor < 75) return false;
            if (cudaStreamCreateWithFlags(&rt.stream, cudaStreamNonBlocking) != cudaSuccess) return false;
            rt.network.blas_workspace_bytes        = 64u * 1024u * 1024u;
            rt.network.density.input_width         = kDensityInputDim;
            rt.network.density.width               = kDensityWidth;
            rt.network.density.output_width        = kDensityOutputDim;
            rt.network.density.hidden_matmuls      = kDensityHiddenLayers - 1u;
            rt.network.density.params_f32.count    = static_cast<std::uint64_t>(kDensityWidth) * kDensityInputDim + static_cast<std::uint64_t>(rt.network.density.hidden_matmuls) * kDensityWidth * kDensityWidth + static_cast<std::uint64_t>(kDensityOutputDim) * kDensityWidth;
            rt.network.color.input_width           = kColorInputDim;
            rt.network.color.width                 = kColorWidth;
            rt.network.color.output_width          = kColorOutputPaddedDim;
            rt.network.color.hidden_matmuls        = kColorHiddenLayers - 1u;
            rt.network.color.params_f32.count      = static_cast<std::uint64_t>(kColorWidth) * kColorInputDim + static_cast<std::uint64_t>(rt.network.color.hidden_matmuls) * kColorWidth * kColorWidth + static_cast<std::uint64_t>(kColorOutputPaddedDim) * kColorWidth;
            rt.network.density.params.count        = rt.network.density.params_f32.count;
            rt.network.density.gradients.count     = rt.network.density.params_f32.count;
            rt.network.density.gradients_tmp.count = rt.network.density.params_f32.count;
            rt.network.density.adam_m.count        = rt.network.density.params_f32.count;
            rt.network.density.adam_v.count        = rt.network.density.params_f32.count;
            rt.network.color.params.count          = rt.network.color.params_f32.count;
            rt.network.color.gradients.count       = rt.network.color.params_f32.count;
            rt.network.color.gradients_tmp.count   = rt.network.color.params_f32.count;
            rt.network.color.adam_m.count          = rt.network.color.params_f32.count;
            rt.network.color.adam_v.count          = rt.network.color.params_f32.count;
            rt.network.density.params_f32.bytes    = rt.network.density.params_f32.count * sizeof(float);
            rt.network.density.params.bytes        = rt.network.density.params.count * sizeof(__half);
            rt.network.density.gradients.bytes     = rt.network.density.gradients.count * sizeof(__half);
            rt.network.density.gradients_tmp.bytes = rt.network.density.gradients_tmp.count * sizeof(__half);
            rt.network.density.adam_m.bytes        = rt.network.density.adam_m.count * sizeof(float);
            rt.network.density.adam_v.bytes        = rt.network.density.adam_v.count * sizeof(float);
            rt.network.color.params_f32.bytes      = rt.network.color.params_f32.count * sizeof(float);
            rt.network.color.params.bytes          = rt.network.color.params.count * sizeof(__half);
            rt.network.color.gradients.bytes       = rt.network.color.gradients.count * sizeof(__half);
            rt.network.color.gradients_tmp.bytes   = rt.network.color.gradients_tmp.count * sizeof(__half);
            rt.network.color.adam_m.bytes          = rt.network.color.adam_m.count * sizeof(float);
            rt.network.color.adam_v.bytes          = rt.network.color.adam_v.count * sizeof(float);
            if (cudaMalloc(&rt.network.density.params_f32.ptr, rt.network.density.params_f32.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.density.params.ptr, rt.network.density.params.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.density.gradients.ptr, rt.network.density.gradients.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.density.gradients_tmp.ptr, rt.network.density.gradients_tmp.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.density.adam_m.ptr, rt.network.density.adam_m.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.density.adam_v.ptr, rt.network.density.adam_v.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.color.params_f32.ptr, rt.network.color.params_f32.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.color.params.ptr, rt.network.color.params.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.color.gradients.ptr, rt.network.color.gradients.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.color.gradients_tmp.ptr, rt.network.color.gradients_tmp.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.color.adam_m.ptr, rt.network.color.adam_m.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.color.adam_v.ptr, rt.network.color.adam_v.bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.network.blas_workspace, rt.network.blas_workspace_bytes) != cudaSuccess) return false;
            if (cudaMalloc(&rt.device_state, sizeof(TrainingDeviceState)) != cudaSuccess) return false;
            std::vector<float> density_host_params(rt.network.density.params_f32.count);
            std::uint64_t density_rng_state = 0x1234ull + 0x9e3779b97f4a7c15ull ^ 0x5678ull * 0xbf58476d1ce4e5b9ull;
            std::uint64_t density_offset    = 0u;
            for (std::uint32_t matrix_idx = 0u; matrix_idx < rt.network.density.hidden_matmuls + 2u; ++matrix_idx) {
                const std::uint32_t fan_in  = matrix_idx == 0u ? kDensityInputDim : kDensityWidth;
                const std::uint32_t fan_out = matrix_idx == rt.network.density.hidden_matmuls + 1u ? kDensityOutputDim : kDensityWidth;
                const float scale           = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
                const std::uint64_t count   = static_cast<std::uint64_t>(fan_out) * fan_in;
                for (std::uint64_t index = 0u; index < count; ++index) {
                    density_rng_state += 0x9e3779b97f4a7c15ull;
                    std::uint64_t z = density_rng_state;
                    z               = (z ^ z >> 30u) * 0xbf58476d1ce4e5b9ull;
                    z               = (z ^ z >> 27u) * 0x94d049bb133111ebull;
                    z ^= z >> 31u;
                    density_host_params[density_offset + index] = (static_cast<float>(static_cast<std::uint32_t>(z >> 32u) >> 8u) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * scale;
                }
                density_offset += count;
            }
            if (cudaMemcpy(rt.network.density.params_f32.ptr, density_host_params.data(), rt.network.density.params_f32.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
            std::vector<float> color_host_params(rt.network.color.params_f32.count);
            std::uint64_t color_rng_state = 0x9abcull + 0x9e3779b97f4a7c15ull ^ 0xdef0ull * 0xbf58476d1ce4e5b9ull;
            std::uint64_t color_offset    = 0u;
            for (std::uint32_t matrix_idx = 0u; matrix_idx < rt.network.color.hidden_matmuls + 2u; ++matrix_idx) {
                const std::uint32_t fan_in  = matrix_idx == 0u ? kColorInputDim : kColorWidth;
                const std::uint32_t fan_out = matrix_idx == rt.network.color.hidden_matmuls + 1u ? kColorOutputPaddedDim : kColorWidth;
                const float scale           = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
                const std::uint64_t count   = static_cast<std::uint64_t>(fan_out) * fan_in;
                for (std::uint64_t index = 0u; index < count; ++index) {
                    color_rng_state += 0x9e3779b97f4a7c15ull;
                    std::uint64_t z = color_rng_state;
                    z               = (z ^ z >> 30u) * 0xbf58476d1ce4e5b9ull;
                    z               = (z ^ z >> 27u) * 0x94d049bb133111ebull;
                    z ^= z >> 31u;
                    color_host_params[color_offset + index] = (static_cast<float>(static_cast<std::uint32_t>(z >> 32u) >> 8u) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * scale;
                }
                color_offset += count;
            }
            if (cudaMemcpy(rt.network.color.params_f32.ptr, color_host_params.data(), rt.network.color.params_f32.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
            constexpr std::uint32_t convert_threads = 256u;
            k_float_to_half<<<(static_cast<std::uint32_t>(rt.network.density.params.count) + convert_threads - 1u) / convert_threads, convert_threads, 0, rt.stream>>>(rt.network.density.params_f32.ptr, rt.network.density.params.ptr, static_cast<std::uint32_t>(rt.network.density.params.count));
            k_float_to_half<<<(static_cast<std::uint32_t>(rt.network.color.params.count) + convert_threads - 1u) / convert_threads, convert_threads, 0, rt.stream>>>(rt.network.color.params_f32.ptr, rt.network.color.params.ptr, static_cast<std::uint32_t>(rt.network.color.params.count));
            if (cudaGetLastError() != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.density.gradients.ptr, 0, rt.network.density.gradients.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.density.gradients_tmp.ptr, 0, rt.network.density.gradients_tmp.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.density.adam_m.ptr, 0, rt.network.density.adam_m.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.density.adam_v.ptr, 0, rt.network.density.adam_v.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.color.gradients.ptr, 0, rt.network.color.gradients.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.color.gradients_tmp.ptr, 0, rt.network.color.gradients_tmp.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.color.adam_m.ptr, 0, rt.network.color.adam_m.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.network.color.adam_v.ptr, 0, rt.network.color.adam_v.bytes, rt.stream) != cudaSuccess) return false;
            if (cudaMemsetAsync(rt.device_state, 0, sizeof(TrainingDeviceState), rt.stream) != cudaSuccess) return false;
            if (!alloc_workspace(rt.workspace, 65536u + kNetworkBatchGranularity)) return false;
            if (!nerf::network::init_network_module(rt.network, rt.stream)) return false;
            if (cudaStreamSynchronize(rt.stream) != cudaSuccess) return false;
            return true;
        } catch (...) {
            destroy_runtime(rt);
            return false;
        }
    }
    static bool render_inference(TrainRuntime& runtime, const ContextStorage& ctx, const std::uint32_t camera_idx, const std::uint32_t samples_per_ray, std::uint32_t* out_rgba) {
        if (!ctx.xforms.ptr || !out_rgba) return false;
        if (samples_per_ray == 0u) return false;
        const std::uint32_t image_width  = ctx.dataset_info.image_width;
        const std::uint32_t image_height = ctx.dataset_info.image_height;
        if (image_width == 0u || image_height == 0u) return true;

        const std::uint64_t total_rays64 = static_cast<std::uint64_t>(image_width) * static_cast<std::uint64_t>(image_height);
        if (total_rays64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) return false;
        const std::uint32_t total_rays = static_cast<std::uint32_t>(total_rays64);

        const std::uint32_t max_chunk_rays = runtime.workspace.rows_capacity / samples_per_ray;
        if (max_chunk_rays == 0u) return false;

        float3 aabb_min = make_float3(0.0f, 0.0f, 0.0f);
        float3 aabb_max = make_float3(1.0f, 1.0f, 1.0f);
        if (runtime.training_configured) {
            aabb_min = make_float3(runtime.training_config.aabb_min_x, runtime.training_config.aabb_min_y, runtime.training_config.aabb_min_z);
            aabb_max = make_float3(runtime.training_config.aabb_max_x, runtime.training_config.aabb_max_y, runtime.training_config.aabb_max_z);
        }

        const auto* cams_4x4_packed = reinterpret_cast<const float*>(ctx.xforms.ptr);
        for (std::uint32_t ray_start = 0u; ray_start < total_rays; ray_start += max_chunk_rays) {
            const std::uint32_t ray_count = std::min(max_chunk_rays, total_rays - ray_start);
            const std::uint32_t rows      = ray_count * samples_per_ray;
            if (!nerf::sampler::write_inference_inputs(runtime.stream, cams_4x4_packed, camera_idx, image_width, image_height, ctx.dataset_info.fx, ctx.dataset_info.fy, ctx.dataset_info.cx, ctx.dataset_info.cy, ray_start, ray_count, samples_per_ray, aabb_min, aabb_max, runtime.workspace.inputs_tmp, runtime.workspace.ray_counts_tmp)) return false;
            if (!nerf::encoder::run_encoder_module(runtime.stream, runtime.workspace.inputs_tmp, rows, runtime.workspace.enc_pts, runtime.workspace.enc_dir)) return false;
            if (!nerf::network::run_network_inference(runtime.network, runtime.workspace, runtime.stream,
                    nerf::network::NetworkInferenceRequest{
                        .encoded_pts = runtime.workspace.enc_pts,
                        .encoded_dir = runtime.workspace.enc_dir,
                        .rows        = rows,
                        .raw_rgb     = runtime.workspace.raw_rgb,
                        .raw_sigma   = runtime.workspace.raw_sigma,
                    }))
                return false;
            if (!nerf::sampler::write_inference_rgba(runtime.stream, runtime.workspace.raw_rgb, runtime.workspace.raw_sigma, runtime.workspace.inputs_tmp, runtime.workspace.ray_counts_tmp, ray_start, ray_count, samples_per_ray, image_width, image_height, out_rgba)) return false;
        }
        return true;
    }
} // namespace nerf::runtime

static void reset_scene_views(nerf::runtime::ContextStorage* ctx) {
    if (!ctx) return;
    ctx->images          = {};
    ctx->xforms          = {};
    ctx->inference_rgba8 = {};
}

static NerfStatus release_scene_storage(nerf::runtime::ContextStorage* ctx) {
    if (!ctx) return NERF_STATUS_INVALID_USAGE;
    if (ctx->scene_device_base && cudaFree(ctx->scene_device_base) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    ctx->scene_device_base = nullptr;
    reset_scene_views(ctx);
    return NERF_STATUS_OK;
}


static NerfStatus upload_dataset_from_create_desc(nerf::runtime::ContextStorage* ctx, const NerfCreateDesc& desc) {
    if (!ctx) return NERF_STATUS_INVALID_USAGE;
    {
        std::scoped_lock lock(ctx->cuda_context->train_runtime_mutex);
        std::shared_ptr<nerf::runtime::TrainRuntime> runtime = ctx->cuda_context->train_runtime;
        runtime->training_configured                         = false;
        runtime->host_generation                             = 0u;
    }
    const NerfStatus release_status = release_scene_storage(ctx);
    if (release_status != NERF_STATUS_OK) return release_status;
    std::uint64_t cursor = 0u;
    nerf::runtime::Region images{};
    nerf::runtime::Region xforms{};
    nerf::runtime::Region inference_rgba8{};
    {
        const std::uint64_t align_mask      = ctx->arena_alignment_bytes - 1u;
        const std::uint64_t inference_bytes = static_cast<std::uint64_t>(desc.image_width) * desc.image_height * 4u;
        cursor                              = (cursor + align_mask) & ~align_mask;
        images.offset_bytes                 = cursor;
        images.size_bytes                   = desc.images_bytes;
        cursor += images.size_bytes;
        cursor              = (cursor + align_mask) & ~align_mask;
        xforms.offset_bytes = cursor;
        xforms.size_bytes   = desc.cameras_bytes;
        cursor += xforms.size_bytes;
        cursor                       = (cursor + align_mask) & ~align_mask;
        inference_rgba8.offset_bytes = cursor;
        inference_rgba8.size_bytes   = inference_bytes;
        cursor += inference_rgba8.size_bytes;
    }
    void* scene_ptr                = nullptr;
    const cudaError_t alloc_status = cudaMalloc(&scene_ptr, cursor);
    if (alloc_status != cudaSuccess) {
        return NERF_STATUS_CUDA_ERROR;
    }
    ctx->scene_device_base          = static_cast<std::byte*>(scene_ptr);
    ctx->images.ptr                 = ctx->scene_device_base + images.offset_bytes;
    ctx->images.size_bytes          = images.size_bytes;
    ctx->xforms.ptr                 = ctx->scene_device_base + xforms.offset_bytes;
    ctx->xforms.size_bytes          = xforms.size_bytes;
    ctx->inference_rgba8.ptr        = ctx->scene_device_base + inference_rgba8.offset_bytes;
    ctx->inference_rgba8.size_bytes = inference_rgba8.size_bytes;
    if (cudaMemcpy(ctx->images.ptr, desc.images_rgba8, ctx->images.size_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) release_scene_storage(ctx);
        return NERF_STATUS_CUDA_ERROR;
    }
    if (cudaMemcpy(ctx->xforms.ptr, desc.cameras_4x4_packed, ctx->xforms.size_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) release_scene_storage(ctx);
        return NERF_STATUS_CUDA_ERROR;
    }
    if (cudaMemset(ctx->inference_rgba8.ptr, 0, ctx->inference_rgba8.size_bytes) != cudaSuccess) {
        (void) release_scene_storage(ctx);
        return NERF_STATUS_CUDA_ERROR;
    }
    ctx->dataset_info.image_count  = desc.image_count;
    ctx->dataset_info.image_width  = desc.image_width;
    ctx->dataset_info.image_height = desc.image_height;
    ctx->dataset_info.fx           = desc.fx;
    ctx->dataset_info.fy           = desc.fy;
    ctx->dataset_info.cx           = desc.cx;
    ctx->dataset_info.cy           = desc.cy;
    return NERF_STATUS_OK;
}


extern "C" {
NerfStatus nerf_create_context(const NerfCreateDesc* desc, void** out_context) {
    if (!desc || !out_context) return NERF_STATUS_INVALID_USAGE;
    *out_context                    = nullptr;
    const NerfCreateDesc create_desc = *desc;
    std::uint64_t cursor             = 0u;
    nerf::runtime::Region occupancy_bitfield{};
    nerf::runtime::Region occupancy_density{};
    {
        const std::uint64_t align_mask     = create_desc.arena_alignment_bytes - 1u;
        const std::uint64_t cell_count     = static_cast<std::uint64_t>(create_desc.occupancy_grid_res) * create_desc.occupancy_grid_res * create_desc.occupancy_grid_res;
        const std::uint64_t bitfield_bytes = (cell_count + 31u) / 32u * sizeof(std::uint32_t);
        const std::uint64_t density_bytes  = cell_count * sizeof(float);
        cursor                             = (cursor + align_mask) & ~align_mask;
        occupancy_bitfield.offset_bytes    = cursor;
        occupancy_bitfield.size_bytes      = bitfield_bytes;
        cursor += bitfield_bytes;
        cursor                         = (cursor + align_mask) & ~align_mask;
        occupancy_density.offset_bytes = cursor;
        occupancy_density.size_bytes   = density_bytes;
        cursor += density_bytes;
    }
    nerf::runtime::Region sample_rays{};
    nerf::runtime::Region sample_steps{};
    nerf::runtime::Region sample_batch_state{};
    {
        const std::uint64_t align_mask        = create_desc.arena_alignment_bytes - 1u;
        const std::uint64_t sample_ray_bytes  = static_cast<std::uint64_t>(create_desc.max_batch_rays) * sizeof(nerf::sampler::SampleRay);
        const std::uint64_t sample_step_bytes = static_cast<std::uint64_t>(create_desc.max_sample_steps) * sizeof(nerf::sampler::SampleStep);
        cursor                                = (cursor + align_mask) & ~align_mask;
        sample_rays.offset_bytes              = cursor;
        sample_rays.size_bytes                = sample_ray_bytes;
        cursor += sample_ray_bytes;
        cursor                    = (cursor + align_mask) & ~align_mask;
        sample_steps.offset_bytes = cursor;
        sample_steps.size_bytes   = sample_step_bytes;
        cursor += sample_step_bytes;
        cursor                          = (cursor + align_mask) & ~align_mask;
        sample_batch_state.offset_bytes = cursor;
        sample_batch_state.size_bytes   = sizeof(nerf::sampler::SampleBatchState);
        cursor += sample_batch_state.size_bytes;
    }
    std::unique_ptr<nerf::runtime::DeviceContext> cuda_context = std::make_unique<nerf::runtime::DeviceContext>();
    std::unique_ptr<nerf::runtime::ContextStorage> context     = std::make_unique<nerf::runtime::ContextStorage>();
    context->cuda_context                                      = cuda_context.release();
    context->occupancy_grid_res                                = create_desc.occupancy_grid_res;
    context->max_sample_steps                                  = create_desc.max_sample_steps;
    context->max_batch_rays                                    = create_desc.max_batch_rays;
    context->arena_alignment_bytes                             = create_desc.arena_alignment_bytes;
    if (cursor != 0u) {
        void* scratch_ptr              = nullptr;
        const cudaError_t alloc_status = cudaMalloc(&scratch_ptr, cursor);
        if (alloc_status != cudaSuccess) {
            delete context->cuda_context;
            return NERF_STATUS_CUDA_ERROR;
        }
        context->scratch_device_base           = static_cast<std::byte*>(scratch_ptr);
        context->occupancy_bitfield.ptr        = context->scratch_device_base + occupancy_bitfield.offset_bytes;
        context->occupancy_bitfield.size_bytes = occupancy_bitfield.size_bytes;
        context->occupancy_density.ptr         = context->scratch_device_base + occupancy_density.offset_bytes;
        context->occupancy_density.size_bytes  = occupancy_density.size_bytes;
        context->sample_rays.ptr               = context->scratch_device_base + sample_rays.offset_bytes;
        context->sample_rays.size_bytes        = sample_rays.size_bytes;
        context->sample_steps.ptr              = context->scratch_device_base + sample_steps.offset_bytes;
        context->sample_steps.size_bytes       = sample_steps.size_bytes;
        context->sample_batch_state.ptr        = context->scratch_device_base + sample_batch_state.offset_bytes;
        context->sample_batch_state.size_bytes = sample_batch_state.size_bytes;
    }
    {
        auto runtime = std::make_shared<nerf::runtime::TrainRuntime>();
        if (!nerf::runtime::init_runtime(*runtime)) {
            nerf::runtime::destroy_runtime(*runtime);
            if (context->scratch_device_base) cudaFree(context->scratch_device_base);
            delete context->cuda_context;
            return NERF_STATUS_FAILURE;
        }
        std::scoped_lock lock(context->cuda_context->train_runtime_mutex);
        context->cuda_context->train_runtime = std::move(runtime);
    }
    const NerfStatus upload_status = upload_dataset_from_create_desc(context.get(), create_desc);
    if (upload_status != NERF_STATUS_OK) {
        std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
        {
            std::scoped_lock lock(context->cuda_context->train_runtime_mutex);
            runtime = std::move(context->cuda_context->train_runtime);
        }
        if (runtime) {
            std::scoped_lock run_lock(runtime->run_mutex);
            nerf::runtime::destroy_runtime(*runtime);
        }
        if (context->scene_device_base) {
            (void) cudaFree(context->scene_device_base);
            context->scene_device_base = nullptr;
        }
        if (context->scratch_device_base) {
            (void) cudaFree(context->scratch_device_base);
            context->scratch_device_base = nullptr;
        }
        delete context->cuda_context;
        context->cuda_context = nullptr;
        return upload_status;
    }
    *out_context = context.release();
    return NERF_STATUS_OK;
}
NerfStatus nerf_destroy_context(void* context) {
    if (!context) return NERF_STATUS_OK;
    const std::unique_ptr<nerf::runtime::ContextStorage> owned{static_cast<nerf::runtime::ContextStorage*>(context)};
    NerfStatus status = NERF_STATUS_OK;
    if (owned->cuda_context) {
        std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
        {
            std::scoped_lock lock(owned->cuda_context->train_runtime_mutex);
            runtime = std::move(owned->cuda_context->train_runtime);
        }
        if (runtime) {
            std::scoped_lock run_lock(runtime->run_mutex);
            nerf::runtime::destroy_runtime(*runtime);
        }
        delete owned->cuda_context;
        owned->cuda_context = nullptr;
    }
    if (owned->scene_device_base) {
        if (cudaFree(owned->scene_device_base) != cudaSuccess && status == NERF_STATUS_OK) status = NERF_STATUS_CUDA_ERROR;
        owned->scene_device_base = nullptr;
    }
    if (owned->scratch_device_base) {
        if (cudaFree(owned->scratch_device_base) != cudaSuccess && status == NERF_STATUS_OK) status = NERF_STATUS_CUDA_ERROR;
        owned->scratch_device_base = nullptr;
    }
    return status;
}
NerfStatus nerf_configure_training(void* context, const NerfTrainingConfig* config) {
    if (!context || !config) return NERF_STATUS_INVALID_USAGE;
    nerf::runtime::ContextStorage* ctx = static_cast<nerf::runtime::ContextStorage*>(context);
    std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
    {
        std::scoped_lock lock(ctx->cuda_context->train_runtime_mutex);
        if (!ctx->scene_device_base) return NERF_STATUS_STATE_ERROR;
        runtime = ctx->cuda_context->train_runtime;
    }
    if (!runtime) return NERF_STATUS_STATE_ERROR;

    const NerfTrainingConfig requested_config = *config;
    const std::uint64_t total_sample_steps    = static_cast<std::uint64_t>(requested_config.rays_per_batch) * static_cast<std::uint64_t>(requested_config.max_sample_steps_per_ray);
    if (requested_config.rays_per_batch == 0u || requested_config.max_sample_steps_per_ray == 0u || requested_config.max_sample_steps_per_ray > 256u) return NERF_STATUS_INVALID_USAGE;
    if (requested_config.occupancy_params.update_interval == 0u || ctx->dataset_info.image_count == 0u) return NERF_STATUS_INVALID_USAGE;
    if (requested_config.rays_per_batch > ctx->max_batch_rays || total_sample_steps > ctx->max_sample_steps || total_sample_steps > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) return NERF_STATUS_INVALID_USAGE;
    const std::uint32_t occupancy_cell_count         = static_cast<std::uint32_t>(static_cast<std::uint64_t>(ctx->occupancy_grid_res) * static_cast<std::uint64_t>(ctx->occupancy_grid_res) * static_cast<std::uint64_t>(ctx->occupancy_grid_res));
    const std::uint32_t occupancy_update_count       = std::min<std::uint32_t>(requested_config.occupancy_params.cells_per_update, occupancy_cell_count);
    const std::uint32_t occupancy_update_rows_padded = (occupancy_update_count + nerf::compile_time::kNetworkBatchGranularity - 1u) / nerf::compile_time::kNetworkBatchGranularity * nerf::compile_time::kNetworkBatchGranularity;
    const nerf::runtime::OccupancyUpdateRequest occupancy_request{
        .bitfield           = reinterpret_cast<std::uint32_t*>(ctx->occupancy_bitfield.ptr),
        .bitfield_bytes     = ctx->occupancy_bitfield.size_bytes,
        .density_grid       = reinterpret_cast<float*>(ctx->occupancy_density.ptr),
        .grid_res           = ctx->occupancy_grid_res,
        .cell_count         = occupancy_cell_count,
        .update_count       = occupancy_update_count,
        .update_rows_padded = occupancy_update_rows_padded,
        .decay              = requested_config.occupancy_params.decay,
        .threshold          = requested_config.occupancy_params.threshold,
        .update_interval    = requested_config.occupancy_params.update_interval,
        .warmup_steps       = requested_config.occupancy_params.warmup_steps,
        .aabb_min           = float3{requested_config.aabb_min_x, requested_config.aabb_min_y, requested_config.aabb_min_z},
        .aabb_max           = float3{requested_config.aabb_max_x, requested_config.aabb_max_y, requested_config.aabb_max_z},
    };
    const nerf::sampler::SamplerRequest sampler_request{
        .stream                   = runtime->stream,
        .device_state             = runtime->device_state,
        .cams_4x4_packed          = reinterpret_cast<const float*>(ctx->xforms.ptr),
        .images                   = reinterpret_cast<const std::uint8_t*>(ctx->images.ptr),
        .bitfield                 = reinterpret_cast<const std::uint32_t*>(ctx->occupancy_bitfield.ptr),
        .sample_rays              = reinterpret_cast<nerf::sampler::SampleRay*>(ctx->sample_rays.ptr),
        .sample_steps             = reinterpret_cast<nerf::sampler::SampleStep*>(ctx->sample_steps.ptr),
        .batch_state              = reinterpret_cast<nerf::sampler::SampleBatchState*>(ctx->sample_batch_state.ptr),
        .occupancy_grid_res       = ctx->occupancy_grid_res,
        .rays_per_batch           = requested_config.rays_per_batch,
        .max_sample_steps_per_ray = requested_config.max_sample_steps_per_ray,
        .image_width              = ctx->dataset_info.image_width,
        .image_height             = ctx->dataset_info.image_height,
        .fx                       = ctx->dataset_info.fx,
        .fy                       = ctx->dataset_info.fy,
        .cx                       = ctx->dataset_info.cx,
        .cy                       = ctx->dataset_info.cy,
        .aabb_min                 = float3{requested_config.aabb_min_x, requested_config.aabb_min_y, requested_config.aabb_min_z},
        .aabb_max                 = float3{requested_config.aabb_max_x, requested_config.aabb_max_y, requested_config.aabb_max_z},
    };
    const nerf::runtime::TrainingGraphConfig training_graph_config{
        .sample_rays      = reinterpret_cast<const nerf::sampler::SampleRay*>(ctx->sample_rays.ptr),
        .sample_steps     = reinterpret_cast<const nerf::sampler::SampleStep*>(ctx->sample_steps.ptr),
        .batch_state      = reinterpret_cast<const nerf::sampler::SampleBatchState*>(ctx->sample_batch_state.ptr),
        .image_count      = ctx->dataset_info.image_count,
        .max_ray_count    = requested_config.rays_per_batch,
        .max_sample_count = static_cast<std::uint32_t>(total_sample_steps),
        .train_cfg =
            nerf::runtime::TrainStepConfig{
                .learning_rate          = requested_config.hyper_params.learning_rate,
                .adam_beta1             = requested_config.hyper_params.adam_beta1,
                .adam_beta2             = requested_config.hyper_params.adam_beta2,
                .adam_eps               = requested_config.hyper_params.adam_eps,
                .grad_clip_norm         = requested_config.hyper_params.grad_clip_norm,
                .update_guard_grad_norm = requested_config.hyper_params.update_guard_grad_norm,
                .loss_scale             = requested_config.hyper_params.loss_scale,
                .lr_decay_ksteps        = requested_config.hyper_params.lr_decay_ksteps,
            },
    };
    {
        std::scoped_lock run_lock(runtime->run_mutex);
        if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        const std::uint32_t occupancy_rows = occupancy_request.update_rows_padded;
        const std::uint32_t workspace_rows = std::max(training_graph_config.max_sample_count + nerf::compile_time::kNetworkBatchGranularity, occupancy_rows);
        if (!nerf::runtime::alloc_workspace(runtime->workspace, workspace_rows)) return NERF_STATUS_FAILURE;
        if (cudaMemsetAsync(runtime->device_state, 0, sizeof(nerf::runtime::TrainingDeviceState), runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(ctx->occupancy_bitfield.ptr, 0, ctx->occupancy_bitfield.size_bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(ctx->occupancy_density.ptr, 0, ctx->occupancy_density.size_bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.density.gradients.ptr, 0, runtime->network.density.gradients.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.density.gradients_tmp.ptr, 0, runtime->network.density.gradients_tmp.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.density.adam_m.ptr, 0, runtime->network.density.adam_m.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.density.adam_v.ptr, 0, runtime->network.density.adam_v.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.color.gradients.ptr, 0, runtime->network.color.gradients.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.color.gradients_tmp.ptr, 0, runtime->network.color.gradients_tmp.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.color.adam_m.ptr, 0, runtime->network.color.adam_m.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaMemsetAsync(runtime->network.color.adam_v.ptr, 0, runtime->network.color.adam_v.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (runtime->train_step_graph_exec) {
            if (cudaGraphExecDestroy(runtime->train_step_graph_exec) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
            runtime->train_step_graph_exec = nullptr;
        }
        if (runtime->train_step_graph) {
            if (cudaGraphDestroy(runtime->train_step_graph) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
            runtime->train_step_graph = nullptr;
        }
        runtime->host_generation                  = 0u;
        runtime->training_config                  = requested_config;
        const std::uint32_t cell_count            = occupancy_request.cell_count;
        const std::uint32_t word_count            = static_cast<std::uint32_t>((occupancy_request.bitfield_bytes + sizeof(std::uint32_t) - 1u) / sizeof(std::uint32_t));
        const std::uint32_t update_rows           = occupancy_request.update_rows_padded;
        constexpr std::uint32_t block_x               = 256u;
        constexpr std::uint32_t training_bucket_rows  = 512u;
        const std::uint32_t training_bucket_count = std::max<std::uint32_t>(1u, (training_graph_config.max_sample_count + training_bucket_rows - 1u) / training_bucket_rows);
        const dim3 full_grid((cell_count + block_x - 1u) / block_x);
        const dim3 word_grid((word_count + block_x - 1u) / block_x);
        const dim3 update_grid((update_rows + block_x - 1u) / block_x);
        constexpr std::uint32_t threads            = 256u;
        const std::uint32_t density_n          = static_cast<std::uint32_t>(runtime->network.density.gradients.count);
        const std::uint32_t color_n            = static_cast<std::uint32_t>(runtime->network.color.gradients.count);
        int capture_device                     = 0;
        cudaExecutionContext_t capture_context = nullptr;
        cudaGraph_t train_step_graph           = nullptr;
        cudaGraph_t captured_graph             = nullptr;
        cudaGraphConditionalHandle training_bucket_handle{};
        cudaGraphNode_t training_node = nullptr;
        cudaError_t graph_error       = cudaSuccess;
        bool capture_ok               = cudaGraphCreate(&train_step_graph, 0) == cudaSuccess;
        runtime->training_configured  = false;
        if (capture_ok) {
            graph_error = cudaGetDevice(&capture_device);
            capture_ok  = graph_error == cudaSuccess;
        }
        if (capture_ok) {
            graph_error = cudaDeviceGetExecutionCtx(&capture_context, capture_device);
            capture_ok  = graph_error == cudaSuccess && capture_context != nullptr;
        }
        if (capture_ok) {
            graph_error = cudaGraphConditionalHandleCreate_v2(&training_bucket_handle, train_step_graph, capture_context, 0u, cudaGraphCondAssignDefault);
            capture_ok  = graph_error == cudaSuccess;
        }
        std::vector<cudaGraphNode_t> prefix_dependencies{};
        bool prefix_capture_open = false;
        if (capture_ok) {
            graph_error         = cudaStreamBeginCaptureToGraph(runtime->stream, train_step_graph, nullptr, nullptr, 0u, cudaStreamCaptureModeRelaxed);
            prefix_capture_open = graph_error == cudaSuccess;
            capture_ok          = prefix_capture_open;
        }
        if (capture_ok) {
            nerf::runtime::k_begin_training_step<<<1, 1, 0, runtime->stream>>>(runtime->device_state, training_graph_config.image_count, occupancy_request.warmup_steps, occupancy_request.update_interval);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok) {
            nerf::runtime::k_update_occupancy_density<<<full_grid, block_x, 0, runtime->stream>>>(occupancy_request.density_grid, cell_count, occupancy_request.decay, runtime->device_state);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok) {
            nerf::runtime::k_update_occupancy_bitfield<<<word_grid, block_x, 0, runtime->stream>>>(occupancy_request.bitfield, word_count, runtime->device_state);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok && update_rows != 0u) {
            nerf::runtime::k_build_occ_inputs<<<update_grid, block_x, 0, runtime->stream>>>(runtime->device_state, update_rows, occupancy_request.update_count, occupancy_request.cell_count, occupancy_request.grid_res, occupancy_request.aabb_min, occupancy_request.aabb_max, runtime->workspace.inputs_tmp);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok && update_rows != 0u) capture_ok = nerf::encoder::run_position_encoder_module(runtime->stream, runtime->workspace.inputs_tmp, update_rows, runtime->workspace.enc_pts);
        if (capture_ok && update_rows != 0u) capture_ok = nerf::network::run_density_inference(runtime->network, runtime->workspace, runtime->stream, runtime->workspace.enc_pts, update_rows, runtime->workspace.raw_sigma);
        if (capture_ok && update_rows != 0u) {
            nerf::runtime::k_update_density_from_sigma<<<update_grid, block_x, 0, runtime->stream>>>(occupancy_request.density_grid, runtime->workspace.raw_sigma, runtime->device_state, occupancy_request.update_count, occupancy_request.cell_count);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok) {
            nerf::runtime::k_rebuild_occ_from_density<<<full_grid, block_x, 0, runtime->stream>>>(occupancy_request.density_grid, runtime->device_state, occupancy_request.cell_count, occupancy_request.threshold, occupancy_request.bitfield);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok) capture_ok = nerf::sampler::run_sampler(sampler_request);
        if (capture_ok && cudaMemsetAsync(runtime->network.density.gradients.ptr, 0, runtime->network.density.gradients.bytes, runtime->stream) != cudaSuccess) capture_ok = false;
        if (capture_ok && cudaMemsetAsync(runtime->network.color.gradients.ptr, 0, runtime->network.color.gradients.bytes, runtime->stream) != cudaSuccess) capture_ok = false;
        if (capture_ok && cudaMemsetAsync(runtime->workspace.loss_sum, 0, sizeof(float), runtime->stream) != cudaSuccess) capture_ok = false;
        if (capture_ok && cudaMemsetAsync(runtime->workspace.grad_sumsq, 0, sizeof(float), runtime->stream) != cudaSuccess) capture_ok = false;
        if (capture_ok && cudaMemsetAsync(runtime->workspace.nonfinite_flag, 0, sizeof(std::uint32_t), runtime->stream) != cudaSuccess) capture_ok = false;
        if (capture_ok) {
            nerf::runtime::k_select_training_bucket<<<1, 1, 0, runtime->stream>>>(training_bucket_handle, training_graph_config.batch_state, training_bucket_rows, training_bucket_count);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok) {
            cudaStreamCaptureStatus capture_status      = cudaStreamCaptureStatusNone;
            const cudaGraphNode_t* capture_dependencies = nullptr;
            size_t capture_dependency_count             = 0u;
            graph_error                                 = cudaStreamGetCaptureInfo(runtime->stream, &capture_status, nullptr, nullptr, &capture_dependencies, nullptr, &capture_dependency_count);
            capture_ok                                  = graph_error == cudaSuccess && capture_status == cudaStreamCaptureStatusActive && capture_dependencies != nullptr && capture_dependency_count != 0u;
            if (capture_ok) prefix_dependencies.assign(capture_dependencies, capture_dependencies + capture_dependency_count);
        }
        if (prefix_capture_open) {
            graph_error = cudaStreamEndCapture(runtime->stream, &captured_graph);
            if (graph_error != cudaSuccess) capture_ok = false;
        }
        if (capture_ok) capture_ok = captured_graph == train_step_graph;
        cudaGraphNodeParams training_node_params{};
        if (capture_ok) {
            training_node_params.type               = cudaGraphNodeTypeConditional;
            training_node_params.conditional.handle = training_bucket_handle;
            training_node_params.conditional.type   = cudaGraphCondTypeSwitch;
            training_node_params.conditional.size   = training_bucket_count + 1u;
            training_node_params.conditional.ctx    = capture_context;
            graph_error                             = cudaGraphAddNode(&training_node, train_step_graph, prefix_dependencies.data(), nullptr, prefix_dependencies.size(), &training_node_params);
            capture_ok                              = graph_error == cudaSuccess;
        }
        if (capture_ok) {
            cudaGraphNode_t zero_bucket_node = nullptr;
            graph_error                      = cudaGraphAddEmptyNode(&zero_bucket_node, training_node_params.conditional.phGraph_out[0], nullptr, 0u);
            capture_ok                       = graph_error == cudaSuccess;
        }
        if (capture_ok) {
            for (std::uint32_t bucket_index = 1u; bucket_index <= training_bucket_count; ++bucket_index) {
                graph_error              = cudaStreamBeginCaptureToGraph(runtime->stream, training_node_params.conditional.phGraph_out[bucket_index], nullptr, nullptr, 0u, cudaStreamCaptureModeRelaxed);
                bool bucket_capture_open = graph_error == cudaSuccess;
                capture_ok               = bucket_capture_open;
                if (capture_ok) {
                    const std::uint32_t bucket_sample_count = bucket_index == training_bucket_count ? training_graph_config.max_sample_count : bucket_index * training_bucket_rows;
                    capture_ok                                 = nerf::network::run_network_training(runtime->network, runtime->workspace, runtime->stream,
                                                        nerf::network::NetworkTrainingRequest{
                                                            .sample_rays      = training_graph_config.sample_rays,
                                                            .sample_steps     = training_graph_config.sample_steps,
                                                            .batch_state      = training_graph_config.batch_state,
                                                            .max_ray_count    = training_graph_config.max_ray_count,
                                                            .max_sample_count = bucket_sample_count,
                        },
                                                        training_graph_config.train_cfg.loss_scale);
                }
                if (bucket_capture_open) {
                    graph_error = cudaStreamEndCapture(runtime->stream, &captured_graph);
                    if (graph_error != cudaSuccess) capture_ok = false;
                }
                if (!capture_ok || captured_graph != training_node_params.conditional.phGraph_out[bucket_index]) break;
            }
        }
        bool tail_capture_open = false;
        if (capture_ok) {
            graph_error       = cudaStreamBeginCaptureToGraph(runtime->stream, train_step_graph, &training_node, nullptr, 1u, cudaStreamCaptureModeRelaxed);
            tail_capture_open = graph_error == cudaSuccess;
            capture_ok        = tail_capture_open;
        }
        if (capture_ok && density_n != 0u) nerf::runtime::k_accum_grad_stats_half<<<(density_n + threads - 1u) / threads, threads, 0, runtime->stream>>>(runtime->network.density.gradients.ptr, density_n, runtime->workspace.grad_sumsq, runtime->workspace.nonfinite_flag);
        if (capture_ok && color_n != 0u) nerf::runtime::k_accum_grad_stats_half<<<(color_n + threads - 1u) / threads, threads, 0, runtime->stream>>>(runtime->network.color.gradients.ptr, color_n, runtime->workspace.grad_sumsq, runtime->workspace.nonfinite_flag);
        if (capture_ok) capture_ok = cudaGetLastError() == cudaSuccess;
        if (capture_ok) {
            nerf::runtime::k_finalize_training_stats<<<1, 1, 0, runtime->stream>>>(runtime->device_state, training_graph_config.batch_state, runtime->workspace.loss_sum, runtime->workspace.grad_sumsq, runtime->workspace.nonfinite_flag, training_graph_config.train_cfg.loss_scale);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok) {
            const float beta1   = training_graph_config.train_cfg.adam_beta1;
            const float beta2   = training_graph_config.train_cfg.adam_beta2;
            const float epsilon = training_graph_config.train_cfg.adam_eps;
            nerf::runtime::k_prepare_adam_step_scalars<<<1, 1, 0, runtime->stream>>>(runtime->device_state, training_graph_config.train_cfg.learning_rate, beta1, beta2, training_graph_config.train_cfg.lr_decay_ksteps, training_graph_config.train_cfg.grad_clip_norm, training_graph_config.train_cfg.update_guard_grad_norm, training_graph_config.train_cfg.loss_scale, runtime->workspace.adam_step_scalars);
            capture_ok = cudaGetLastError() == cudaSuccess;
            if (capture_ok && density_n != 0u) nerf::runtime::k_adam_step_half<<<(density_n + threads - 1u) / threads, threads, 0, runtime->stream>>>(runtime->network.density.params_f32.ptr, runtime->network.density.params.ptr, runtime->network.density.gradients.ptr, runtime->network.density.adam_m.ptr, runtime->network.density.adam_v.ptr, density_n, beta1, beta2, epsilon, runtime->workspace.adam_step_scalars);
            if (capture_ok && color_n != 0u) nerf::runtime::k_adam_step_half<<<(color_n + threads - 1u) / threads, threads, 0, runtime->stream>>>(runtime->network.color.params_f32.ptr, runtime->network.color.params.ptr, runtime->network.color.gradients.ptr, runtime->network.color.adam_m.ptr, runtime->network.color.adam_v.ptr, color_n, beta1, beta2, epsilon, runtime->workspace.adam_step_scalars);
            if (capture_ok) capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (capture_ok) {
            nerf::runtime::k_commit_training_step<<<1, 1, 0, runtime->stream>>>(runtime->device_state);
            capture_ok = cudaGetLastError() == cudaSuccess;
        }
        if (tail_capture_open) {
            graph_error = cudaStreamEndCapture(runtime->stream, &captured_graph);
            if (graph_error != cudaSuccess) capture_ok = false;
        }
        if (capture_ok) capture_ok = captured_graph == train_step_graph;
        if (capture_ok) {
            graph_error = cudaGraphInstantiate(&runtime->train_step_graph_exec, train_step_graph, nullptr, nullptr, 0);
            capture_ok  = graph_error == cudaSuccess;
        }
        if (!capture_ok) {
            if (runtime->train_step_graph_exec) {
                (void) cudaGraphExecDestroy(runtime->train_step_graph_exec);
                runtime->train_step_graph_exec = nullptr;
            }
            if (train_step_graph) (void) cudaGraphDestroy(train_step_graph);
            return NERF_STATUS_CUDA_ERROR;
        }
        runtime->train_step_graph = train_step_graph;
        if (cudaGraphUpload(runtime->train_step_graph_exec, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
        runtime->training_configured = true;
    }
    return NERF_STATUS_OK;
}
NerfStatus nerf_train_step(void* context) {
    if (!context) return NERF_STATUS_INVALID_USAGE;
    const nerf::runtime::ContextStorage* ctx = static_cast<nerf::runtime::ContextStorage*>(context);
    std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
    {
        std::scoped_lock lock(ctx->cuda_context->train_runtime_mutex);
        if (!ctx->scene_device_base) return NERF_STATUS_STATE_ERROR;
        runtime = ctx->cuda_context->train_runtime;
    }
    if (!runtime || !runtime->training_configured) return NERF_STATUS_STATE_ERROR;
    std::scoped_lock run_lock(runtime->run_mutex);
    if (!runtime->train_step_graph_exec) return NERF_STATUS_STATE_ERROR;
    if (cudaGraphLaunch(runtime->train_step_graph_exec, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    ++runtime->host_generation;
    return NERF_STATUS_OK;
}
NerfStatus nerf_read_train_stats(void* context, NerfTrainStats* out_stats) {
    if (!context || !out_stats) return NERF_STATUS_INVALID_USAGE;
    const nerf::runtime::ContextStorage* ctx = static_cast<nerf::runtime::ContextStorage*>(context);
    std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
    {
        std::scoped_lock lock(ctx->cuda_context->train_runtime_mutex);
        runtime = ctx->cuda_context->train_runtime;
    }
    if (!runtime || !runtime->training_configured) return NERF_STATUS_STATE_ERROR;
    std::scoped_lock run_lock(runtime->run_mutex);
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemcpy(out_stats, &runtime->device_state->stats, sizeof(NerfTrainStats), cudaMemcpyDeviceToHost) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    runtime->host_generation = out_stats->completed_steps;
    return NERF_STATUS_OK;
}
NerfStatus nerf_inference(void* context, const NerfInferenceRequest* request, NerfInferenceInfo* out_info) {
    if (!context || !request) return NERF_STATUS_INVALID_USAGE;
    const nerf::runtime::ContextStorage* ctx = static_cast<nerf::runtime::ContextStorage*>(context);
    if (!ctx->scene_device_base || !ctx->inference_rgba8.ptr) return NERF_STATUS_STATE_ERROR;
    if (request->samples_per_ray == 0u) return NERF_STATUS_INVALID_USAGE;
    const std::uint32_t samples_per_ray = request->samples_per_ray;
    const std::uint64_t expected_bytes  = static_cast<std::uint64_t>(ctx->dataset_info.image_width) * static_cast<std::uint64_t>(ctx->dataset_info.image_height) * 4u;
    std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
    {
        std::scoped_lock lock(ctx->cuda_context->train_runtime_mutex);
        runtime = ctx->cuda_context->train_runtime;
    }
    if (!runtime) return NERF_STATUS_STATE_ERROR;
    std::scoped_lock run_lock(runtime->run_mutex);
    if (!nerf::runtime::render_inference(*runtime, *ctx, request->camera_idx, samples_per_ray, reinterpret_cast<std::uint32_t*>(ctx->inference_rgba8.ptr))) return NERF_STATUS_CUDA_ERROR;
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (request->dst_rgba8 != nullptr) {
        const cudaMemcpyKind kind = request->memory_kind == NERF_MEMORY_CUDA_DEVICE ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
        if (cudaMemcpy(request->dst_rgba8, ctx->inference_rgba8.ptr, expected_bytes, kind) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    }
    if (out_info) {
        out_info->capacity_bytes   = ctx->inference_rgba8.size_bytes;
        out_info->valid_bytes      = expected_bytes;
        out_info->row_stride_bytes = ctx->dataset_info.image_width * 4u;
        out_info->width            = ctx->dataset_info.image_width;
        out_info->height           = ctx->dataset_info.image_height;
        out_info->generation       = runtime->host_generation;
    }
    return NERF_STATUS_OK;
}
NerfStatus nerf_save_network_weights(void* context, const NerfCheckpointFileDesc* desc) {
    if (!context || !desc || !desc->path_utf8) return NERF_STATUS_INVALID_USAGE;
    const nerf::runtime::ContextStorage* ctx = static_cast<nerf::runtime::ContextStorage*>(context);
    std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
    {
        std::scoped_lock lock(ctx->cuda_context->train_runtime_mutex);
        runtime = ctx->cuda_context->train_runtime;
    }
    if (!runtime) return NERF_STATUS_STATE_ERROR;
    nerf::network::NetworkCheckpointLayout layout{};
    if (!nerf::network::describe_network_checkpoint_layout(runtime->network, layout)) return NERF_STATUS_FAILURE;
    std::scoped_lock run_lock(runtime->run_mutex);
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    nerf::host::HostCheckpointData checkpoint{};
    checkpoint.density_params_f32.resize(runtime->network.density.params_f32.count);
    checkpoint.color_params_f32.resize(runtime->network.color.params_f32.count);
    if (cudaMemcpy(checkpoint.density_params_f32.data(), runtime->network.density.params_f32.ptr, runtime->network.density.params_f32.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemcpy(checkpoint.color_params_f32.data(), runtime->network.color.params_f32.ptr, runtime->network.color.params_f32.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    std::ofstream file(desc->path_utf8, std::ios::binary | std::ios::trunc);
    if (!file) return NERF_STATUS_IO_ERROR;
    nlohmann::json header  = nlohmann::json::object();
    header["__metadata__"] = {
        {"format", "nerf/network-v1"},
        {"density_input_dim", layout.density_input_width},
        {"density_hidden_layers", layout.density_hidden_layers},
        {"density_width", layout.density_width},
        {"density_output_dim", layout.density_output_width},
        {"color_input_dim", layout.color_input_width},
        {"color_hidden_layers", layout.color_hidden_layers},
        {"color_width", layout.color_width},
        {"color_output_dim", layout.color_output_width},
    };
    std::uint64_t data_offset = 0u;
    for (std::uint32_t index = 0u; index < layout.tensor_count; ++index) {
        const nerf::network::NetworkCheckpointTensorLayout& tensor = layout.tensors[index];
        const std::uint64_t element_count                          = static_cast<std::uint64_t>(tensor.rows) * tensor.cols;
        const std::uint64_t begin                                  = data_offset;
        const std::uint64_t end                                    = begin + element_count * sizeof(float);
        header[std::string{tensor.name}]                           = {
            {"dtype", "F32"},
            {"shape", {tensor.rows, tensor.cols}},
            {"data_offsets", {begin, end}},
        };
        data_offset = end;
    }
    const std::string header_bytes  = header.dump();
    const std::uint64_t header_size = header_bytes.size();
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    file.write(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));
    if (!file) return NERF_STATUS_IO_ERROR;
    for (std::uint32_t index = 0u; index < layout.tensor_count; ++index) {
        const nerf::network::NetworkCheckpointTensorLayout& tensor = layout.tensors[index];
        const float* src                                           = tensor.network_index == 0u ? checkpoint.density_params_f32.data() + tensor.offset : checkpoint.color_params_f32.data() + tensor.offset;
        const std::uint64_t element_count                          = static_cast<std::uint64_t>(tensor.rows) * tensor.cols;
        file.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(element_count * sizeof(float)));
        if (!file) return NERF_STATUS_IO_ERROR;
    }
    if (!file.good()) return NERF_STATUS_IO_ERROR;
    return NERF_STATUS_OK;
}
NerfStatus nerf_load_network_weights(void* context, const NerfCheckpointFileDesc* desc) {
    if (!context || !desc || !desc->path_utf8) return NERF_STATUS_INVALID_USAGE;
    const nerf::runtime::ContextStorage* ctx = static_cast<nerf::runtime::ContextStorage*>(context);
    std::shared_ptr<nerf::runtime::TrainRuntime> runtime;
    {
        std::scoped_lock lock(ctx->cuda_context->train_runtime_mutex);
        runtime = ctx->cuda_context->train_runtime;
    }
    if (!runtime) return NERF_STATUS_STATE_ERROR;
    nerf::network::NetworkCheckpointLayout layout{};
    if (!nerf::network::describe_network_checkpoint_layout(runtime->network, layout)) return NERF_STATUS_FAILURE;
    nerf::host::HostCheckpointData checkpoint{};
    try {
        std::ifstream file(desc->path_utf8, std::ios::binary);
        if (!file) return NERF_STATUS_IO_ERROR;

        std::uint64_t header_size = 0u;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        if (!file) return NERF_STATUS_IO_ERROR;

        std::string header_bytes(header_size, '\0');
        file.read(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));
        if (!file) return NERF_STATUS_IO_ERROR;

        const nlohmann::json header   = nlohmann::json::parse(header_bytes);
        const std::uint64_t data_base = sizeof(std::uint64_t) + header_size;

        checkpoint.density_params_f32.resize(runtime->network.density.params_f32.count);
        checkpoint.color_params_f32.resize(runtime->network.color.params_f32.count);
        for (std::uint32_t index = 0u; index < layout.tensor_count; ++index) {
            const nerf::network::NetworkCheckpointTensorLayout& tensor_layout = layout.tensors[index];
            const nlohmann::json& tensor                                      = header.at(std::string{tensor_layout.name});
            const std::uint64_t begin                                         = tensor.at("data_offsets").at(0).get<std::uint64_t>();
            const std::uint64_t element_count                                 = static_cast<std::uint64_t>(tensor_layout.rows) * tensor_layout.cols;
            file.seekg(static_cast<std::streamoff>(data_base + begin), std::ios::beg);
            if (tensor_layout.network_index == 0u)
                file.read(reinterpret_cast<char*>(checkpoint.density_params_f32.data() + tensor_layout.offset), static_cast<std::streamsize>(element_count * sizeof(float)));
            else
                file.read(reinterpret_cast<char*>(checkpoint.color_params_f32.data() + tensor_layout.offset), static_cast<std::streamsize>(element_count * sizeof(float)));
            if (!file) return NERF_STATUS_IO_ERROR;
        }
    } catch (...) {
        return NERF_STATUS_IO_ERROR;
    }
    std::scoped_lock run_lock(runtime->run_mutex);
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemcpy(runtime->network.density.params_f32.ptr, checkpoint.density_params_f32.data(), runtime->network.density.params_f32.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemcpy(runtime->network.color.params_f32.ptr, checkpoint.color_params_f32.data(), runtime->network.color.params_f32.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    constexpr std::uint32_t convert_threads = 256u;
    nerf::runtime::k_float_to_half<<<(static_cast<std::uint32_t>(runtime->network.density.params.count) + convert_threads - 1u) / convert_threads, convert_threads, 0, runtime->stream>>>(runtime->network.density.params_f32.ptr, runtime->network.density.params.ptr, static_cast<std::uint32_t>(runtime->network.density.params.count));
    nerf::runtime::k_float_to_half<<<(static_cast<std::uint32_t>(runtime->network.color.params.count) + convert_threads - 1u) / convert_threads, convert_threads, 0, runtime->stream>>>(runtime->network.color.params_f32.ptr, runtime->network.color.params.ptr, static_cast<std::uint32_t>(runtime->network.color.params.count));
    if (cudaGetLastError() != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.density.gradients.ptr, 0, runtime->network.density.gradients.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.density.gradients_tmp.ptr, 0, runtime->network.density.gradients_tmp.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.density.adam_m.ptr, 0, runtime->network.density.adam_m.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.density.adam_v.ptr, 0, runtime->network.density.adam_v.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.color.gradients.ptr, 0, runtime->network.color.gradients.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.color.gradients_tmp.ptr, 0, runtime->network.color.gradients_tmp.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.color.adam_m.ptr, 0, runtime->network.color.adam_m.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->network.color.adam_v.ptr, 0, runtime->network.color.adam_v.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(runtime->device_state, 0, sizeof(nerf::runtime::TrainingDeviceState), runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(ctx->occupancy_bitfield.ptr, 0, ctx->occupancy_bitfield.size_bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(ctx->occupancy_density.ptr, 0, ctx->occupancy_density.size_bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_ERROR;
    runtime->host_generation = 0u;
    return NERF_STATUS_OK;
}
}
