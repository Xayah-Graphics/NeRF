#include "nerf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mma.h>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>
#include <vector_types.h>

#include "json/json.hpp"

constexpr std::uint32_t kMaxSampleStepsPerRay = 256u;

constexpr std::uint32_t kPosFreqs = 10u;
constexpr std::uint32_t kDirFreqs = 4u;
constexpr std::uint32_t kPtsInDim = 3u + 2u * 3u * kPosFreqs;
constexpr std::uint32_t kDirInDim = 3u + 2u * 3u * kDirFreqs;

constexpr std::uint32_t kTrainChunkRows              = 65536u;
constexpr std::uint32_t kSamplerBlockRays            = 128u;
constexpr std::uint32_t kSamplerMaxSampleStepsPerRay = 256u;

constexpr std::uint32_t kDensityInputDim             = 64u;
constexpr std::uint32_t kDensityOutputDim            = 16u;
constexpr std::uint32_t kDensityWidth                = 128u;
constexpr std::uint32_t kDensityHiddenLayers         = 5u;
constexpr std::uint32_t kGeoFeatureDim               = 15u;
constexpr std::uint32_t kColorInputDim               = 48u;
constexpr std::uint32_t kColorOutputDim              = 3u;
constexpr std::uint32_t kColorOutputPaddedDim        = 16u;
constexpr std::uint32_t kColorWidth                  = 128u;
constexpr std::uint32_t kColorHiddenLayers           = 3u;
constexpr std::uint32_t kNetworkBatchGranularity     = 256u;
constexpr std::uint32_t kNetworkCheckpointMaxTensors = 32u;

constexpr std::uint32_t kFusedWidth             = 128u;
constexpr std::uint32_t kFusedOutputWidth       = 16u;
constexpr std::uint32_t kFusedSkew              = 8u;
constexpr std::uint32_t kFusedInputSkew         = 8u;
constexpr std::uint32_t kFusedBlockRows         = kFusedWidth / 16u;
constexpr std::uint32_t kFusedIters             = 8u;
constexpr std::uint32_t kFusedBatchQuantum      = 16u * kFusedIters;
constexpr std::size_t kFusedForwardShmemDensity = std::max(sizeof(__half) * (kFusedWidth + 16u) * (kDensityInputDim + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));
constexpr std::size_t kFusedForwardShmemColor   = std::max(sizeof(__half) * (kFusedWidth + 16u) * (kColorInputDim + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));
constexpr std::size_t kFusedBackwardShmem       = sizeof(__half) * 16u * kFusedIters * (kFusedWidth + kFusedSkew);
constexpr std::uint32_t kFusedElemsPerLoad      = kFusedBlockRows * 32u * 8u;
constexpr std::uint32_t kFusedWeightsStride     = kFusedWidth * kFusedWidth;

constexpr std::uint32_t kWmmaThreadsX      = 32u;
constexpr std::uint32_t kWmmaThreadsY      = 8u;
constexpr std::uint32_t kWmmaThreadsZ      = 1u;
constexpr std::uint32_t kInputGradThreadsX = 16u;
constexpr std::uint32_t kInputGradThreadsY = 16u;
constexpr std::uint32_t kInputGradThreadsZ = 1u;
constexpr std::uint32_t kThreads256        = 256u;
constexpr std::uint64_t kArenaAlignBytes   = 16u;
constexpr std::uint32_t kOccupancyBlockX   = 256u;
constexpr std::uint32_t kConvertThreads    = 256u;

constexpr float kNetworkLossScale    = 128.0f;
constexpr float kSamplerEps          = 1e-8f;
constexpr float kInv255              = 1.0f / 255.0f;
constexpr float kBlasAlpha           = 1.0f;
constexpr float kBlasBeta            = 0.0f;
constexpr float kGlobalGradClipNorm  = 1.0f;
constexpr float kUpdateGuardGradNorm = 100.0f;
constexpr float kInvLossScale        = 1.0f / kNetworkLossScale;

namespace nerf::host {
    struct HostDatasetData {
        std::vector<std::uint8_t> images_rgba8{};
        std::vector<float> c2w_4x4{};
        NerfDatasetInfo info{};
    };

    struct HostCheckpointData {
        std::vector<float> density_params_f32{};
        std::vector<float> color_params_f32{};
    };
} // namespace nerf::host


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
        std::uint32_t overflow          = 0u;
    };

    struct SamplerRequest {
        cudaStream_t stream                    = nullptr;
        const float4* cams                     = nullptr;
        const std::uint8_t* images             = nullptr;
        const std::uint32_t* bitfield          = nullptr;
        SampleRay* sample_rays                 = nullptr;
        SampleStep* sample_steps               = nullptr;
        SampleBatchState* batch_state          = nullptr;
        std::uint32_t frame_index              = 0u;
        std::uint32_t camera_idx               = 0u;
        std::uint32_t occupancy_grid_res       = 0u;
        std::uint32_t rays_per_batch           = 0u;
        std::uint32_t max_sample_steps_per_ray = 0u;
        std::uint32_t max_sample_step_count    = 0u;
        std::uint32_t image_width              = 0u;
        std::uint32_t image_height             = 0u;
        float3 aabb_min{};
        float3 aabb_max{};
    };

} // namespace nerf::sampler


namespace nerf::network {
    template <typename T>
    struct DeviceBuffer {
        T* ptr              = nullptr;
        std::uint64_t count = 0u;
        std::size_t bytes   = 0u;
    };

    struct NetworkWorkspace {
        std::uint32_t rows_capacity     = 0u;
        unsigned char* arena            = nullptr;
        float* inputs_tmp               = nullptr;
        float* enc_pts                  = nullptr;
        float* enc_dir                  = nullptr;
        float* raw_rgb                  = nullptr;
        float* d_rgb                    = nullptr;
        float* raw_sigma                = nullptr;
        float* d_sigma                  = nullptr;
        float* trans_tmp                = nullptr;
        float* loss_sum                 = nullptr;
        float* grad_sumsq               = nullptr;
        std::uint32_t* nonfinite_flag   = nullptr;
        __half* density_input           = nullptr;
        __half* density_output          = nullptr;
        __half* density_doutput         = nullptr;
        __half* density_forward_hidden  = nullptr;
        __half* density_backward_hidden = nullptr;
        __half* color_input             = nullptr;
        __half* color_output            = nullptr;
        __half* color_doutput           = nullptr;
        __half* color_dinput            = nullptr;
        __half* color_forward_hidden    = nullptr;
        __half* color_backward_hidden   = nullptr;
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
        void* blas_handle = nullptr;
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
        std::uint32_t ray_count                            = 0u;
        std::uint32_t sample_count                         = 0u;
    };

    struct NetworkCheckpointTensorLayout {
        char name[64]{};
        std::uint64_t offset        = 0u;
        std::uint32_t rows          = 0u;
        std::uint32_t cols          = 0u;
        std::uint32_t network_index = 0u;
    };

    struct NetworkCheckpointLayout {
        NetworkCheckpointTensorLayout tensors[kNetworkCheckpointMaxTensors]{};
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


namespace nerf::io {
    static_assert(std::endian::native == std::endian::little);

    NerfStatus load_dataset_host_data(const NerfDatasetLoadDesc& desc, nerf::host::HostDatasetData& out_data) {
        if (!desc.path_utf8) return NERF_STATUS_INVALID_ARGUMENT;
        if (!std::isfinite(desc.scale) || !(desc.scale > 0.0f)) return NERF_STATUS_INVALID_ARGUMENT;
        if (!std::isfinite(desc.offset.x) || !std::isfinite(desc.offset.y) || !std::isfinite(desc.offset.z)) return NERF_STATUS_INVALID_ARGUMENT;

        const NerfCoordBasis basis_list[] = {
            desc.source_system.world,
            desc.source_system.camera,
            desc.target_system.world,
            desc.target_system.camera,
        };
        for (const NerfCoordBasis& basis : basis_list) {
            const std::uint32_t dirs[] = {basis.x, basis.y, basis.z};
            std::uint32_t seen_axes    = 0u;
            for (const std::uint32_t dir : dirs) {
                std::uint32_t axis = 0u;
                switch (dir) {
                case NERF_COORD_AXIS_POSITIVE_X:
                case NERF_COORD_AXIS_NEGATIVE_X: axis = 0u; break;
                case NERF_COORD_AXIS_POSITIVE_Y:
                case NERF_COORD_AXIS_NEGATIVE_Y: axis = 1u; break;
                case NERF_COORD_AXIS_POSITIVE_Z:
                case NERF_COORD_AXIS_NEGATIVE_Z: axis = 2u; break;
                default: return NERF_STATUS_INVALID_ARGUMENT;
                }
                if ((seen_axes & 1u << axis) != 0u) return NERF_STATUS_INVALID_ARGUMENT;
                seen_axes |= 1u << axis;
            }
            if (seen_axes != 0x7u) return NERF_STATUS_INVALID_ARGUMENT;
        }

        std::filesystem::path root{desc.path_utf8};
        std::filesystem::path json_path = root;
        if (!json_path.has_extension() || json_path.extension() != ".json") json_path = root / "transforms_train.json";

        std::ifstream json_file(json_path, std::ios::binary);
        if (!json_file) return NERF_STATUS_FILE_NOT_FOUND;

        nlohmann::json json_value;
        try {
            json_file >> json_value;
        } catch (...) {
            return NERF_STATUS_JSON_PARSE_FAILED;
        }

        if (!json_value.contains("camera_angle_x")) return NERF_STATUS_MISSING_REQUIRED_FIELD;
        if (!json_value.contains("frames")) return NERF_STATUS_MISSING_REQUIRED_FIELD;

        float camera_angle_x = 0.0f;
        try {
            camera_angle_x = json_value.at("camera_angle_x").get<float>();
        } catch (...) {
            return NERF_STATUS_MISSING_REQUIRED_FIELD;
        }
        if (!std::isfinite(camera_angle_x)) return NERF_STATUS_INVALID_TRANSFORM_MATRIX;

        if (!json_value.at("frames").is_array() || json_value.at("frames").empty()) return NERF_STATUS_MISSING_REQUIRED_FIELD;

        const nlohmann::json& frames  = json_value.at("frames");
        const std::size_t frame_count = frames.size();
        if (frame_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return NERF_STATUS_OVERFLOW;

        std::vector<std::filesystem::path> image_paths(frame_count);
        std::vector<float> row_major_4x4(frame_count * 16u, 0.0f);

        for (std::size_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
            const nlohmann::json& frame = frames[frame_index];
            if (!frame.contains("file_path")) return NERF_STATUS_MISSING_REQUIRED_FIELD;
            if (!frame.contains("transform_matrix")) return NERF_STATUS_MISSING_REQUIRED_FIELD;

            try {
                std::filesystem::path image_path = json_path.parent_path() / frame.at("file_path").get<std::string>();
                if (!image_path.has_extension()) image_path.replace_extension(".png");
                std::error_code ec;
                image_path = std::filesystem::absolute(image_path, ec);
                if (ec) return NERF_STATUS_FILE_NOT_FOUND;
                image_path = std::filesystem::weakly_canonical(image_path, ec);
                if (ec) return NERF_STATUS_FILE_NOT_FOUND;
                image_paths[frame_index] = std::move(image_path);
            } catch (...) {
                return NERF_STATUS_MISSING_REQUIRED_FIELD;
            }

            try {
                for (std::size_t r = 0u; r < 4u; ++r) {
                    for (std::size_t c = 0u; c < 4u; ++c) {
                        const float value = frame.at("transform_matrix").at(r).at(c).get<float>();
                        if (!std::isfinite(value)) return NERF_STATUS_INVALID_TRANSFORM_MATRIX;
                        row_major_4x4[frame_index * 16u + r * 4u + c] = value;
                    }
                }
            } catch (...) {
                return NERF_STATUS_INVALID_TRANSFORM_MATRIX;
            }
        }

        std::error_code image_ec{};
        if (!std::filesystem::is_regular_file(image_paths[0], image_ec) || image_ec) return NERF_STATUS_FILE_NOT_FOUND;

        int first_width       = 0;
        int first_height      = 0;
        int first_channels    = 0;
        stbi_uc* first_pixels = stbi_load(image_paths[0].string().c_str(), &first_width, &first_height, &first_channels, 4);
        if (!first_pixels) return NERF_STATUS_IMAGE_LOAD_FAILED;
        if (first_width <= 0 || first_height <= 0) {
            stbi_image_free(first_pixels);
            return NERF_STATUS_IMAGE_LOAD_FAILED;
        }

        const std::uint64_t image_count_u64 = frame_count;
        std::uint64_t image_bytes           = image_count_u64;
        if (image_bytes > std::numeric_limits<std::uint64_t>::max() / 4u) {
            stbi_image_free(first_pixels);
            return NERF_STATUS_OVERFLOW;
        }
        image_bytes *= 4u;
        if (image_bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(first_width)) {
            stbi_image_free(first_pixels);
            return NERF_STATUS_OVERFLOW;
        }
        image_bytes *= static_cast<std::uint64_t>(first_width);
        if (image_bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(first_height)) {
            stbi_image_free(first_pixels);
            return NERF_STATUS_OVERFLOW;
        }
        image_bytes *= static_cast<std::uint64_t>(first_height);

        const std::uint64_t bytes_per_image = static_cast<std::uint64_t>(first_width) * static_cast<std::uint64_t>(first_height) * 4u;
        const float width_f                 = static_cast<float>(first_width);
        const float height_f                = static_cast<float>(first_height);
        const float fx                      = 0.5f * width_f / std::tan(camera_angle_x * 0.5f);
        const float fy                      = fx;
        const float cx                      = 0.5f * (width_f - 1.0f);
        const float cy                      = 0.5f * (height_f - 1.0f);

        std::vector<std::uint8_t> host_images_rgba8(image_bytes);
        std::memcpy(host_images_rgba8.data(), first_pixels, bytes_per_image);
        stbi_image_free(first_pixels);

        std::atomic<std::size_t> next_image_index{1u};
        std::atomic<int> decode_failure{NERF_STATUS_OK};
        std::size_t worker_count = 0u;
        if (frame_count > 1u) {
            const std::size_t hardware_threads = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
            worker_count                       = std::min<std::size_t>(frame_count - 1u, hardware_threads > 1u ? hardware_threads - 1u : 1u);
        }

        std::vector<std::jthread> workers{};
        workers.reserve(worker_count);
        for (std::size_t worker_index = 0u; worker_index < worker_count; ++worker_index) {
            workers.emplace_back([&host_images_rgba8, &image_paths, &next_image_index, &decode_failure, bytes_per_image, first_width, first_height]() {
                while (true) {
                    if (decode_failure.load(std::memory_order_relaxed) != static_cast<int>(NERF_STATUS_OK)) return;

                    const std::size_t image_index = next_image_index.fetch_add(1u, std::memory_order_relaxed);
                    if (image_index >= image_paths.size()) return;

                    std::error_code local_ec{};
                    if (!std::filesystem::is_regular_file(image_paths[image_index], local_ec) || local_ec) {
                        decode_failure.store(NERF_STATUS_FILE_NOT_FOUND, std::memory_order_relaxed);
                        return;
                    }

                    int width       = 0;
                    int height      = 0;
                    int channels    = 0;
                    stbi_uc* pixels = stbi_load(image_paths[image_index].string().c_str(), &width, &height, &channels, 4);
                    if (!pixels) {
                        decode_failure.store(NERF_STATUS_IMAGE_LOAD_FAILED, std::memory_order_relaxed);
                        return;
                    }
                    if (width != first_width || height != first_height) {
                        stbi_image_free(pixels);
                        decode_failure.store(NERF_STATUS_INCONSISTENT_IMAGE_RESOLUTION, std::memory_order_relaxed);
                        return;
                    }

                    std::memcpy(host_images_rgba8.data() + image_index * bytes_per_image, pixels, bytes_per_image);
                    stbi_image_free(pixels);
                }
            });
        }
        workers.clear();

        if (decode_failure.load(std::memory_order_relaxed) != static_cast<int>(NERF_STATUS_OK)) {
            return static_cast<NerfStatus>(decode_failure.load(std::memory_order_relaxed));
        }

        std::vector<float> host_c2w_4x4(frame_count * 16u, 0.0f);

        float src_world_basis[9]  = {};
        float src_camera_basis[9] = {};
        float dst_world_basis[9]  = {};
        float dst_camera_basis[9] = {};

        const std::uint32_t src_world_dirs[]  = {desc.source_system.world.x, desc.source_system.world.y, desc.source_system.world.z};
        const std::uint32_t src_camera_dirs[] = {desc.source_system.camera.x, desc.source_system.camera.y, desc.source_system.camera.z};
        const std::uint32_t dst_world_dirs[]  = {desc.target_system.world.x, desc.target_system.world.y, desc.target_system.world.z};
        const std::uint32_t dst_camera_dirs[] = {desc.target_system.camera.x, desc.target_system.camera.y, desc.target_system.camera.z};

        for (std::size_t c = 0u; c < 3u; ++c) {
            std::uint32_t src_world_axis  = 0u;
            std::uint32_t src_camera_axis = 0u;
            std::uint32_t dst_world_axis  = 0u;
            std::uint32_t dst_camera_axis = 0u;
            float src_world_sign          = 0.0f;
            float src_camera_sign         = 0.0f;
            float dst_world_sign          = 0.0f;
            float dst_camera_sign         = 0.0f;

            switch (src_world_dirs[c]) {
            case NERF_COORD_AXIS_POSITIVE_X:
                src_world_axis = 0u;
                src_world_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_X:
                src_world_axis = 0u;
                src_world_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Y:
                src_world_axis = 1u;
                src_world_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Y:
                src_world_axis = 1u;
                src_world_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Z:
                src_world_axis = 2u;
                src_world_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Z:
                src_world_axis = 2u;
                src_world_sign = -1.0f;
                break;
            default: return NERF_STATUS_INVALID_ARGUMENT;
            }
            switch (src_camera_dirs[c]) {
            case NERF_COORD_AXIS_POSITIVE_X:
                src_camera_axis = 0u;
                src_camera_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_X:
                src_camera_axis = 0u;
                src_camera_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Y:
                src_camera_axis = 1u;
                src_camera_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Y:
                src_camera_axis = 1u;
                src_camera_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Z:
                src_camera_axis = 2u;
                src_camera_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Z:
                src_camera_axis = 2u;
                src_camera_sign = -1.0f;
                break;
            default: return NERF_STATUS_INVALID_ARGUMENT;
            }
            switch (dst_world_dirs[c]) {
            case NERF_COORD_AXIS_POSITIVE_X:
                dst_world_axis = 0u;
                dst_world_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_X:
                dst_world_axis = 0u;
                dst_world_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Y:
                dst_world_axis = 1u;
                dst_world_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Y:
                dst_world_axis = 1u;
                dst_world_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Z:
                dst_world_axis = 2u;
                dst_world_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Z:
                dst_world_axis = 2u;
                dst_world_sign = -1.0f;
                break;
            default: return NERF_STATUS_INVALID_ARGUMENT;
            }
            switch (dst_camera_dirs[c]) {
            case NERF_COORD_AXIS_POSITIVE_X:
                dst_camera_axis = 0u;
                dst_camera_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_X:
                dst_camera_axis = 0u;
                dst_camera_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Y:
                dst_camera_axis = 1u;
                dst_camera_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Y:
                dst_camera_axis = 1u;
                dst_camera_sign = -1.0f;
                break;
            case NERF_COORD_AXIS_POSITIVE_Z:
                dst_camera_axis = 2u;
                dst_camera_sign = 1.0f;
                break;
            case NERF_COORD_AXIS_NEGATIVE_Z:
                dst_camera_axis = 2u;
                dst_camera_sign = -1.0f;
                break;
            default: return NERF_STATUS_INVALID_ARGUMENT;
            }

            src_world_basis[src_world_axis * 3u + c]   = src_world_sign;
            src_camera_basis[src_camera_axis * 3u + c] = src_camera_sign;
            dst_world_basis[dst_world_axis * 3u + c]   = dst_world_sign;
            dst_camera_basis[dst_camera_axis * 3u + c] = dst_camera_sign;
        }

        float world_map[9]  = {};
        float camera_map[9] = {};
        for (std::size_t r = 0u; r < 3u; ++r) {
            for (std::size_t c = 0u; c < 3u; ++c) {
                for (std::size_t k = 0u; k < 3u; ++k) {
                    world_map[r * 3u + c] += dst_world_basis[k * 3u + r] * src_world_basis[k * 3u + c];
                    camera_map[r * 3u + c] += src_camera_basis[k * 3u + r] * dst_camera_basis[k * 3u + c];
                }
            }
        }

        for (std::size_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
            const float* row_major = row_major_4x4.data() + frame_index * 16u;

            const float src_rot[9] = {
                row_major[0],
                row_major[1],
                row_major[2],
                row_major[4],
                row_major[5],
                row_major[6],
                row_major[8],
                row_major[9],
                row_major[10],
            };
            const float src_translation[3] = {row_major[3], row_major[7], row_major[11]};

            float tmp_rot[9]         = {};
            float dst_rot[9]         = {};
            float dst_translation[3] = {};

            for (std::size_t r = 0u; r < 3u; ++r) {
                for (std::size_t c = 0u; c < 3u; ++c) {
                    for (std::size_t k = 0u; k < 3u; ++k) tmp_rot[r * 3u + c] += world_map[r * 3u + k] * src_rot[k * 3u + c];
                }
            }
            for (std::size_t r = 0u; r < 3u; ++r) {
                for (std::size_t c = 0u; c < 3u; ++c) {
                    for (std::size_t k = 0u; k < 3u; ++k) dst_rot[r * 3u + c] += tmp_rot[r * 3u + k] * camera_map[k * 3u + c];
                }
            }
            for (std::size_t r = 0u; r < 3u; ++r) {
                for (std::size_t k = 0u; k < 3u; ++k) dst_translation[r] += world_map[r * 3u + k] * src_translation[k];
            }

            float* packed = host_c2w_4x4.data() + frame_index * 16u;
            packed[0]     = dst_rot[0];
            packed[1]     = dst_rot[3];
            packed[2]     = dst_rot[6];
            packed[3]     = fx;
            packed[4]     = dst_rot[1];
            packed[5]     = dst_rot[4];
            packed[6]     = dst_rot[7];
            packed[7]     = fy;
            packed[8]     = dst_rot[2];
            packed[9]     = dst_rot[5];
            packed[10]    = dst_rot[8];
            packed[11]    = cx;
            packed[12]    = dst_translation[0] * desc.scale + desc.offset.x;
            packed[13]    = dst_translation[1] * desc.scale + desc.offset.y;
            packed[14]    = dst_translation[2] * desc.scale + desc.offset.z;
            packed[15]    = cy;
        }

        out_data.images_rgba8 = std::move(host_images_rgba8);
        out_data.c2w_4x4      = std::move(host_c2w_4x4);
        out_data.info         = NerfDatasetInfo{
                    .image_count   = static_cast<std::uint32_t>(frame_count),
                    .image_width   = static_cast<std::uint32_t>(first_width),
                    .image_height  = static_cast<std::uint32_t>(first_height),
                    .images_bytes  = image_bytes,
                    .c2w_bytes     = out_data.c2w_4x4.size() * sizeof(float),
                    .fx            = fx,
                    .fy            = fy,
                    .cx            = cx,
                    .cy            = cy,
                    .source_system = desc.source_system,
                    .target_system = desc.target_system,
        };
        return NERF_STATUS_OK;
    }

    NerfStatus save_network_checkpoint_file(const NerfCheckpointFileDesc& desc, const nerf::network::NetworkCheckpointLayout& layout, const nerf::host::HostCheckpointData& data) {
        if (!desc.path_utf8) return NERF_STATUS_INVALID_ARGUMENT;
        if (layout.tensor_count == 0u) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (data.density_params_f32.size() != layout.density_param_count) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (data.color_params_f32.size() != layout.color_param_count) return NERF_STATUS_CHECKPOINT_MISMATCH;

        std::ofstream file(desc.path_utf8, std::ios::binary | std::ios::trunc);
        if (!file) return NERF_STATUS_INTERNAL_ERROR;

        nlohmann::json header  = nlohmann::json::object();
        header["__metadata__"] = {
            {"format", "nerf/network-v1"},
            {"density_input_dim", std::to_string(layout.density_input_width)},
            {"density_hidden_layers", std::to_string(layout.density_hidden_layers)},
            {"density_width", std::to_string(layout.density_width)},
            {"density_output_dim", std::to_string(layout.density_output_width)},
            {"color_input_dim", std::to_string(layout.color_input_width)},
            {"color_hidden_layers", std::to_string(layout.color_hidden_layers)},
            {"color_width", std::to_string(layout.color_width)},
            {"color_output_dim", std::to_string(layout.color_output_width)},
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
        const std::string header_bytes = header.dump();

        const std::uint64_t header_size = header_bytes.size();
        file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        file.write(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));
        if (!file) return NERF_STATUS_INTERNAL_ERROR;

        for (std::uint32_t index = 0u; index < layout.tensor_count; ++index) {
            const nerf::network::NetworkCheckpointTensorLayout& tensor = layout.tensors[index];
            const float* src                                           = tensor.network_index == 0u ? data.density_params_f32.data() + tensor.offset : data.color_params_f32.data() + tensor.offset;
            const std::uint64_t element_count                          = static_cast<std::uint64_t>(tensor.rows) * tensor.cols;
            file.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(element_count * sizeof(float)));
            if (!file) return NERF_STATUS_INTERNAL_ERROR;
        }
        if (!file.good()) return NERF_STATUS_INTERNAL_ERROR;
        return NERF_STATUS_OK;
    }

    NerfStatus load_network_checkpoint_file(const NerfCheckpointFileDesc& desc, const nerf::network::NetworkCheckpointLayout& layout, nerf::host::HostCheckpointData& data) {
        if (!desc.path_utf8) return NERF_STATUS_INVALID_ARGUMENT;
        if (layout.tensor_count == 0u) return NERF_STATUS_CHECKPOINT_MISMATCH;

        std::ifstream file(desc.path_utf8, std::ios::binary);
        if (!file) return NERF_STATUS_FILE_NOT_FOUND;

        file.seekg(0, std::ios::end);
        const std::uint64_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (file_size < sizeof(std::uint64_t)) return NERF_STATUS_CHECKPOINT_INVALID;

        std::uint64_t header_size = 0u;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        if (!file) return NERF_STATUS_CHECKPOINT_INVALID;
        if (header_size == 0u || header_size > file_size - sizeof(std::uint64_t)) return NERF_STATUS_CHECKPOINT_INVALID;

        std::string header_bytes(header_size, '\0');
        file.read(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));
        if (!file) return NERF_STATUS_CHECKPOINT_INVALID;

        nlohmann::json header = nlohmann::json::object();
        try {
            header = nlohmann::json::parse(header_bytes);
        } catch (...) {
            return NERF_STATUS_CHECKPOINT_INVALID;
        }
        if (!header.is_object()) return NERF_STATUS_CHECKPOINT_INVALID;
        if (header.size() != static_cast<std::size_t>(layout.tensor_count) + 1u) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!header.contains("__metadata__")) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!header["__metadata__"].is_object()) return NERF_STATUS_CHECKPOINT_MISMATCH;

        const nlohmann::json& metadata = header["__metadata__"];
        if (metadata.size() != 9u) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("format") || metadata["format"] != "nerf/network-v1") return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("density_input_dim") || metadata["density_input_dim"] != std::to_string(layout.density_input_width)) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("density_hidden_layers") || metadata["density_hidden_layers"] != std::to_string(layout.density_hidden_layers)) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("density_width") || metadata["density_width"] != std::to_string(layout.density_width)) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("density_output_dim") || metadata["density_output_dim"] != std::to_string(layout.density_output_width)) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("color_input_dim") || metadata["color_input_dim"] != std::to_string(layout.color_input_width)) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("color_hidden_layers") || metadata["color_hidden_layers"] != std::to_string(layout.color_hidden_layers)) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("color_width") || metadata["color_width"] != std::to_string(layout.color_width)) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (!metadata.contains("color_output_dim") || metadata["color_output_dim"] != std::to_string(layout.color_output_width)) return NERF_STATUS_CHECKPOINT_MISMATCH;

        const std::uint64_t data_base = sizeof(std::uint64_t) + header_size;
        std::uint64_t expected_offset = 0u;
        std::uint64_t density_count   = 0u;
        std::uint64_t color_count     = 0u;
        for (std::uint32_t index = 0u; index < layout.tensor_count; ++index) {
            const nerf::network::NetworkCheckpointTensorLayout& tensor_layout = layout.tensors[index];
            if (!header.contains(std::string{tensor_layout.name})) return NERF_STATUS_CHECKPOINT_MISMATCH;
            const nlohmann::json& tensor = header[std::string{tensor_layout.name}];
            if (!tensor.is_object()) return NERF_STATUS_CHECKPOINT_MISMATCH;
            if (!tensor.contains("dtype") || tensor["dtype"] != "F32") return NERF_STATUS_CHECKPOINT_MISMATCH;
            if (!tensor.contains("shape") || !tensor["shape"].is_array() || tensor["shape"].size() != 2u) return NERF_STATUS_CHECKPOINT_MISMATCH;
            if (!tensor.contains("data_offsets") || !tensor["data_offsets"].is_array() || tensor["data_offsets"].size() != 2u) return NERF_STATUS_CHECKPOINT_MISMATCH;

            const std::uint64_t rows = tensor["shape"][0].get<std::uint64_t>();
            const std::uint64_t cols = tensor["shape"][1].get<std::uint64_t>();
            if (rows != tensor_layout.rows || cols != tensor_layout.cols) return NERF_STATUS_CHECKPOINT_MISMATCH;

            const std::uint64_t begin = tensor["data_offsets"][0].get<std::uint64_t>();
            const std::uint64_t end   = tensor["data_offsets"][1].get<std::uint64_t>();
            const std::uint64_t bytes = rows * cols * sizeof(float);
            if (begin != expected_offset || end != begin + bytes) return NERF_STATUS_CHECKPOINT_MISMATCH;
            if (data_base + end > file_size) return NERF_STATUS_CHECKPOINT_INVALID;
            expected_offset = end;
            if (tensor_layout.network_index == 0u)
                density_count += rows * cols;
            else
                color_count += rows * cols;
        }
        if (data_base + expected_offset != file_size) return NERF_STATUS_CHECKPOINT_INVALID;
        if (density_count != layout.density_param_count) return NERF_STATUS_CHECKPOINT_MISMATCH;
        if (color_count != layout.color_param_count) return NERF_STATUS_CHECKPOINT_MISMATCH;

        data.density_params_f32.resize(density_count);
        data.color_params_f32.resize(color_count);
        for (std::uint32_t index = 0u; index < layout.tensor_count; ++index) {
            const nerf::network::NetworkCheckpointTensorLayout& tensor_layout = layout.tensors[index];
            const nlohmann::json& tensor                                      = header[std::string{tensor_layout.name}];
            const std::uint64_t rows                                          = tensor["shape"][0].get<std::uint64_t>();
            const std::uint64_t cols                                          = tensor["shape"][1].get<std::uint64_t>();
            const std::uint64_t begin                                         = tensor["data_offsets"][0].get<std::uint64_t>();
            const std::uint64_t elements                                      = rows * cols;
            file.seekg(static_cast<std::streamoff>(data_base + begin), std::ios::beg);
            if (tensor_layout.network_index == 0u) {
                file.read(reinterpret_cast<char*>(data.density_params_f32.data() + tensor_layout.offset), static_cast<std::streamsize>(elements * sizeof(float)));
            } else {
                file.read(reinterpret_cast<char*>(data.color_params_f32.data() + tensor_layout.offset), static_cast<std::streamsize>(elements * sizeof(float)));
            }
            if (!file) return NERF_STATUS_CHECKPOINT_INVALID;
        }
        return NERF_STATUS_OK;
    }
} // namespace nerf::io


namespace nerf::encoder {
    namespace {
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

            float* out_p = enc_pts + static_cast<std::uint64_t>(idx) * kPtsInDim;
            float* out_d = enc_dir + static_cast<std::uint64_t>(idx) * kDirInDim;

            out_p[0] = px;
            out_p[1] = py;
            out_p[2] = pz;

            std::uint32_t p_off = 3u;
            float p_freq        = 1.0f;
            for (std::uint32_t l = 0; l < kPosFreqs; ++l) {
                const float ax = px * p_freq;
                const float ay = py * p_freq;
                const float az = pz * p_freq;
                float sx = 0.0f, cx = 0.0f;
                float sy = 0.0f, cy = 0.0f;
                float sz = 0.0f, cz = 0.0f;
                __sincosf(ax, &sx, &cx);
                __sincosf(ay, &sy, &cy);
                __sincosf(az, &sz, &cz);
                out_p[p_off + 0u] = sx;
                out_p[p_off + 1u] = sy;
                out_p[p_off + 2u] = sz;
                out_p[p_off + 3u] = cx;
                out_p[p_off + 4u] = cy;
                out_p[p_off + 5u] = cz;
                p_off += 6u;
                p_freq *= 2.0f;
            }

            out_d[0] = dx;
            out_d[1] = dy;
            out_d[2] = dz;

            std::uint32_t d_off = 3u;
            float d_freq        = 1.0f;
            for (std::uint32_t l = 0; l < kDirFreqs; ++l) {
                const float ax = dx * d_freq;
                const float ay = dy * d_freq;
                const float az = dz * d_freq;
                float sx = 0.0f, cx = 0.0f;
                float sy = 0.0f, cy = 0.0f;
                float sz = 0.0f, cz = 0.0f;
                __sincosf(ax, &sx, &cx);
                __sincosf(ay, &sy, &cy);
                __sincosf(az, &sz, &cz);
                out_d[d_off + 0u] = sx;
                out_d[d_off + 1u] = sy;
                out_d[d_off + 2u] = sz;
                out_d[d_off + 3u] = cx;
                out_d[d_off + 4u] = cy;
                out_d[d_off + 5u] = cz;
                d_off += 6u;
                d_freq *= 2.0f;
            }
        }

    } // namespace

    bool run_encoder_module(cudaStream_t stream, const float* sample_inputs, const std::uint32_t rows, float* encoded_pts, float* encoded_dir) {
        constexpr std::uint32_t threads = kThreads256;
        const std::uint32_t blocks      = (rows + threads - 1u) / threads;
        k_encode_sample_inputs<<<blocks, threads, 0, stream>>>(sample_inputs, rows, encoded_pts, encoded_dir);
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

            const std::uint32_t ix = min(grid_res - 1u, static_cast<std::uint32_t>(rel.x * static_cast<float>(grid_res)));
            const std::uint32_t iy = min(grid_res - 1u, static_cast<std::uint32_t>(rel.y * static_cast<float>(grid_res)));
            const std::uint32_t iz = min(grid_res - 1u, static_cast<std::uint32_t>(rel.z * static_cast<float>(grid_res)));

            const std::uint32_t idx  = ix + iy * grid_res + iz * grid_res * grid_res;
            const std::uint32_t word = idx >> 5u;
            const std::uint32_t bit  = idx & 31u;
            return (bitfield[word] & 1u << bit) != 0u;
        }

        struct CameraParams {
            float4 c0{};
            float4 c1{};
            float4 c2{};
            float4 c3{};
            float fx = 0.0f;
            float fy = 0.0f;
            float cx = 0.0f;
            float cy = 0.0f;
        };

        __device__ __forceinline__ CameraParams load_camera_params(const float4* cams, const std::uint32_t camera_idx) {
            const std::uint32_t cam_base = camera_idx * 4u;
            CameraParams cam{};
            cam.c0 = cams[cam_base + 0u];
            cam.c1 = cams[cam_base + 1u];
            cam.c2 = cams[cam_base + 2u];
            cam.c3 = cams[cam_base + 3u];
            cam.fx = cam.c0.w;
            cam.fy = cam.c1.w;
            cam.cx = cam.c2.w;
            cam.cy = cam.c3.w;
            return cam;
        }

        __device__ __forceinline__ bool compute_world_ray_dir(const CameraParams& cam, const float pixel_x, const float pixel_y_flipped, float3* out_ray_dir) {
            float3 dir_cam{};
            dir_cam.x = (pixel_x - cam.cx) / cam.fx;
            dir_cam.y = (pixel_y_flipped - cam.cy) / cam.fy;
            dir_cam.z = -1.0f;

            float3 ray_dir{};
            ray_dir.x = cam.c0.x * dir_cam.x + cam.c1.x * dir_cam.y + cam.c2.x * dir_cam.z;
            ray_dir.y = cam.c0.y * dir_cam.x + cam.c1.y * dir_cam.y + cam.c2.y * dir_cam.z;
            ray_dir.z = cam.c0.z * dir_cam.x + cam.c1.z * dir_cam.y + cam.c2.z * dir_cam.z;

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
            constexpr float eps = kSamplerEps;

            float3 inv_dir{};
            inv_dir.x = fabsf(ray_dir.x) > eps ? 1.0f / ray_dir.x : ray_dir.x >= 0.0f ? 1e8f : -1e8f;
            inv_dir.y = fabsf(ray_dir.y) > eps ? 1.0f / ray_dir.y : ray_dir.y >= 0.0f ? 1e8f : -1e8f;
            inv_dir.z = fabsf(ray_dir.z) > eps ? 1.0f / ray_dir.z : ray_dir.z >= 0.0f ? 1e8f : -1e8f;

            const float3 t0    = make_float3((aabb_min.x - ray_origin.x) * inv_dir.x, (aabb_min.y - ray_origin.y) * inv_dir.y, (aabb_min.z - ray_origin.z) * inv_dir.z);
            const float3 t1    = make_float3((aabb_max.x - ray_origin.x) * inv_dir.x, (aabb_max.y - ray_origin.y) * inv_dir.y, (aabb_max.z - ray_origin.z) * inv_dir.z);
            const float3 tminv = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
            const float3 tmaxv = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

            float t_near      = fmaxf(fmaxf(tminv.x, tminv.y), tminv.z);
            const float t_far = fminf(fminf(tmaxv.x, tmaxv.y), tmaxv.z);
            if (!(t_near <= t_far && t_far > 0.0f)) return false;
            if (t_near < 0.0f) t_near = 0.0f;
            *out_t_near = t_near;
            *out_t_far  = t_far;
            return true;
        }

        __global__ void k_begin_sampling_step(SampleBatchState* __restrict__ batch_state) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            batch_state->active_ray_count  = 0u;
            batch_state->sample_step_count = 0u;
            batch_state->overflow          = 0u;
        }

        __global__ void k_sample_rays_flat(const std::uint32_t frame_index, const std::uint32_t camera_idx, const float4* __restrict__ cams, const std::uint8_t* __restrict__ images, const std::uint32_t image_width, const std::uint32_t image_height, const std::uint32_t rays_per_batch, const std::uint32_t max_sample_steps_per_ray, const std::uint32_t max_sample_step_count, const float3 aabb_min, const float3 aabb_max, const std::uint32_t* __restrict__ bitfield,
            const std::uint32_t grid_res, SampleRay* __restrict__ sample_rays, SampleStep* __restrict__ sample_steps, SampleBatchState* __restrict__ batch_state) {
            const std::uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray_idx >= rays_per_batch) return;

            __shared__ CameraParams shared_cam;
            __shared__ std::uint32_t shared_camera_idx;
            __shared__ const std::uint8_t* shared_camera_image;

            const std::uint64_t per_cam_bytes = static_cast<std::uint64_t>(image_width) * static_cast<std::uint64_t>(image_height) * 4ull;
            if (threadIdx.x == 0u) {
                shared_camera_idx   = camera_idx;
                shared_cam          = load_camera_params(cams, shared_camera_idx);
                shared_camera_image = images + static_cast<std::uint64_t>(shared_camera_idx) * per_cam_bytes;
            }
            __syncthreads();

            const std::uint32_t seed    = frame_index * 1315423911u ^ ray_idx * 9781u;
            const float pixel_x         = rand01(seed ^ 0xA511E9B3u) * static_cast<float>(image_width);
            const float pixel_y         = rand01(seed ^ 0x63D83595u) * static_cast<float>(image_height);
            const float pixel_y_flipped = static_cast<float>(image_height) - 1.0f - pixel_y;
            const float3 ray_origin     = make_float3(shared_cam.c3.x, shared_cam.c3.y, shared_cam.c3.z);

            float3 ray_dir{};
            float t_near = 0.0f;
            float t_far  = 0.0f;
            if (!compute_world_ray_dir(shared_cam, pixel_x, pixel_y_flipped, &ray_dir)) return;
            if (!intersect_aabb_ray(ray_origin, ray_dir, aabb_min, aabb_max, &t_near, &t_far)) return;

            const float t_range = t_far - t_near;
            if (!(t_range > 0.0f)) return;

            const float dt_min     = t_range / static_cast<float>(max_sample_steps_per_ray);
            const float cone_angle = cone_angle_from_focal(shared_cam.fx, shared_cam.fy);

            float sample_t_mid[kSamplerMaxSampleStepsPerRay];
            float sample_dt[kSamplerMaxSampleStepsPerRay];
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

            const float px         = fmaxf(0.0f, fminf(pixel_x, static_cast<float>(image_width) - 1.0f));
            const float py         = fmaxf(0.0f, fminf(pixel_y, static_cast<float>(image_height) - 1.0f));
            const int x0           = __float2int_rd(px);
            const int y0           = __float2int_rd(py);
            const int x1           = min(x0 + 1, static_cast<int>(image_width) - 1);
            const int y1           = min(y0 + 1, static_cast<int>(image_height) - 1);
            const float tx         = px - static_cast<float>(x0);
            const float ty         = py - static_cast<float>(y0);
            const int stride       = static_cast<int>(image_width) * 4;
            const int idx00        = y0 * stride + x0 * 4;
            const int idx10        = y0 * stride + x1 * 4;
            const int idx01        = y1 * stride + x0 * 4;
            const int idx11        = y1 * stride + x1 * 4;
            constexpr float inv255 = kInv255;

            const float3 s00 = make_float3(static_cast<float>(shared_camera_image[idx00 + 0]) * inv255, static_cast<float>(shared_camera_image[idx00 + 1]) * inv255, static_cast<float>(shared_camera_image[idx00 + 2]) * inv255);
            const float3 s10 = make_float3(static_cast<float>(shared_camera_image[idx10 + 0]) * inv255, static_cast<float>(shared_camera_image[idx10 + 1]) * inv255, static_cast<float>(shared_camera_image[idx10 + 2]) * inv255);
            const float3 s01 = make_float3(static_cast<float>(shared_camera_image[idx01 + 0]) * inv255, static_cast<float>(shared_camera_image[idx01 + 1]) * inv255, static_cast<float>(shared_camera_image[idx01 + 2]) * inv255);
            const float3 s11 = make_float3(static_cast<float>(shared_camera_image[idx11 + 0]) * inv255, static_cast<float>(shared_camera_image[idx11 + 1]) * inv255, static_cast<float>(shared_camera_image[idx11 + 2]) * inv255);

            const float a00 = static_cast<float>(shared_camera_image[idx00 + 3]) * inv255;
            const float a10 = static_cast<float>(shared_camera_image[idx10 + 3]) * inv255;
            const float a01 = static_cast<float>(shared_camera_image[idx01 + 3]) * inv255;
            const float a11 = static_cast<float>(shared_camera_image[idx11 + 3]) * inv255;

            const float3 a      = make_float3(s00.x * (1.0f - tx) + s10.x * tx, s00.y * (1.0f - tx) + s10.y * tx, s00.z * (1.0f - tx) + s10.z * tx);
            const float3 b      = make_float3(s01.x * (1.0f - tx) + s11.x * tx, s01.y * (1.0f - tx) + s11.y * tx, s01.z * (1.0f - tx) + s11.z * tx);
            const float3 gt_rgb = make_float3(a.x * (1.0f - ty) + b.x * ty, a.y * (1.0f - ty) + b.y * ty, a.z * (1.0f - ty) + b.z * ty);

            const float aa    = a00 * (1.0f - tx) + a10 * tx;
            const float ab    = a01 * (1.0f - tx) + a11 * tx;
            const float alpha = aa * (1.0f - ty) + ab * ty;
            const float3 gt   = make_float3(gt_rgb.x * alpha + (1.0f - alpha), gt_rgb.y * alpha + (1.0f - alpha), gt_rgb.z * alpha + (1.0f - alpha));

            const std::uint32_t active_slot  = atomicAdd(&batch_state->active_ray_count, 1u);
            const std::uint32_t sample_begin = atomicAdd(&batch_state->sample_step_count, sample_count);
            if (active_slot >= rays_per_batch || sample_begin > max_sample_step_count || sample_count > max_sample_step_count - sample_begin) {
                atomicExch(&batch_state->overflow, 1u);
                return;
            }

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
    } // namespace

    bool run_sampler(const SamplerRequest& request) {
        if (!request.stream || !request.cams || !request.images || !request.bitfield) return false;
        if (!request.sample_rays || !request.sample_steps || !request.batch_state) return false;
        if (request.rays_per_batch == 0u || request.max_sample_steps_per_ray == 0u || request.max_sample_step_count == 0u) return false;

        k_begin_sampling_step<<<1, 1, 0, request.stream>>>(request.batch_state);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return false;

        const std::uint32_t blocks = (request.rays_per_batch + kSamplerBlockRays - 1u) / kSamplerBlockRays;
        k_sample_rays_flat<<<blocks, kSamplerBlockRays, 0, request.stream>>>(request.frame_index, request.camera_idx, request.cams, request.images, request.image_width, request.image_height, request.rays_per_batch, request.max_sample_steps_per_ray, request.max_sample_step_count, request.aabb_min, request.aabb_max, request.bitfield, request.occupancy_grid_res, request.sample_rays, request.sample_steps, request.batch_state);
        return cudaGetLastError() == cudaSuccess;
    }

} // namespace nerf::sampler


namespace nerf::network {
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

            const std::uint32_t lane        = threadIdx.x;
            const std::uint32_t warp        = threadIdx.y;
            const std::uint32_t lane_offset = 8u * lane % kFusedWidth;
            const std::uint32_t row         = (8u * lane + warp * 8u * 32u) / kFusedWidth;
            const std::uint32_t weights_col = 16u * warp;
            __half* const weights_shmem     = act_shmem + 16u * (input_width + kFusedInputSkew);

            const std::uint32_t thread_elem_idx    = (lane + warp * 32u) * 8u;
            constexpr std::uint32_t elems_per_load = kFusedElemsPerLoad;
            const std::uint32_t n_weight_elems     = kFusedWidth * input_width;

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

            const std::uint32_t lane        = threadIdx.x;
            const std::uint32_t warp        = threadIdx.y;
            __half* const weights_shmem     = act_shmem + kFusedIters * 16u * (kFusedWidth + kFusedSkew);
            const std::uint32_t weights_row = 8u * lane % kFusedWidth;
            const std::uint32_t weights_col = (8u * lane + 8u * 32u * warp) / kFusedWidth;

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
            constexpr std::uint32_t weights_stride   = kFusedWeightsStride;
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

            const std::uint32_t lane               = threadIdx.x;
            const std::uint32_t warp               = threadIdx.y;
            const std::uint32_t lane_offset        = 8u * lane % kFusedWidth;
            const std::uint32_t row                = (8u * lane + warp * 8u * 32u) / kFusedWidth;
            const std::uint32_t weights_col        = 16u * warp;
            const std::uint32_t elem_idx_base      = 16u * blockIdx.x * kFusedIters;
            constexpr std::uint32_t weights_stride = kFusedWeightsStride;
            const std::uint32_t layer_stride       = kFusedWidth * batch_size;

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
    } // namespace

    bool init_network_module(NetworkSet& network_set, cudaStream_t stream) {
        auto* handle = reinterpret_cast<cublasHandle_t>(network_set.blas_handle);
        if (handle == nullptr) {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
            network_set.blas_handle = handle;
        }
        if (handle == nullptr) return false;
        if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return false;
        if (cudaFuncSetAttribute(k_fully_fused_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, std::max(kFusedForwardShmemDensity, kFusedForwardShmemColor)) != cudaSuccess) return false;
        if (cudaFuncSetAttribute(k_fully_fused_backward, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedBackwardShmem) != cudaSuccess) return false;
        return true;
    }

    bool describe_network_checkpoint_layout(const NetworkSet& network_set, NetworkCheckpointLayout& layout) {
        layout                       = NetworkCheckpointLayout{};
        layout.density_param_count   = network_set.density.params_f32.count;
        layout.color_param_count     = network_set.color.params_f32.count;
        layout.density_input_width   = network_set.density.input_width;
        layout.density_width         = network_set.density.width;
        layout.density_hidden_layers = network_set.density.hidden_matmuls + 1u;
        layout.density_output_width  = network_set.density.output_width;
        layout.color_input_width     = network_set.color.input_width;
        layout.color_width           = network_set.color.width;
        layout.color_hidden_layers   = network_set.color.hidden_matmuls + 1u;
        layout.color_output_width    = network_set.color.output_width;

        std::uint32_t tensor_index         = 0u;
        std::uint64_t density_offset       = 0u;
        std::uint64_t color_offset         = 0u;
        const std::uint32_t density_layers = network_set.density.hidden_matmuls + 2u;
        const std::uint32_t color_layers   = network_set.color.hidden_matmuls + 2u;
        if (density_layers + color_layers > kNetworkCheckpointMaxTensors) return false;

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

        constexpr dim3 threads       = {kWmmaThreadsX, kWmmaThreadsY, kWmmaThreadsZ};
        const dim3 blocks            = {rows / kFusedBatchQuantum, 1u, 1u};
        const std::size_t shmem_size = std::max(sizeof(__half) * (kFusedWidth + 16u) * (network.input_width + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));
        k_fully_fused_forward<<<blocks, threads, shmem_size, stream>>>(input, network.params.ptr, nullptr, output, network.input_width, network.hidden_matmuls);
        return cudaGetLastError() == cudaSuccess;
    }

    static bool fully_fused_mlp_forward(const FusedNetworkState& network, cudaStream_t stream, const __half* input, std::uint32_t rows, __half* output, __half* forward_hidden) {
        if (network.params.ptr == nullptr || input == nullptr || output == nullptr || forward_hidden == nullptr) return false;
        if (rows == 0u || rows % kFusedBatchQuantum != 0u) return false;

        constexpr dim3 threads       = {kWmmaThreadsX, kWmmaThreadsY, kWmmaThreadsZ};
        const dim3 blocks            = {rows / kFusedBatchQuantum, 1u, 1u};
        const std::size_t shmem_size = std::max(sizeof(__half) * (kFusedWidth + 16u) * (network.input_width + kFusedInputSkew), sizeof(__half) * (16u + 16u * kFusedIters) * (kFusedWidth + kFusedSkew));

        k_fully_fused_forward<<<blocks, threads, shmem_size, stream>>>(input, network.params.ptr, forward_hidden, output, network.input_width, network.hidden_matmuls);
        return cudaGetLastError() == cudaSuccess;
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

        if (row < rows && input_col < prefix_width) {
            dinput[static_cast<std::uint64_t>(row) * input_width + input_col] = __float2half_rn(sum);
        }
    }

    static bool fully_fused_mlp_backward(const FusedNetworkState& network, void* blas_handle, cudaStream_t stream, const __half* input, const __half* doutput, std::uint32_t rows, const std::uint32_t dinput_prefix_width, __half* dinput, __half* backward_hidden, const __half* forward_hidden) {
        auto* handle = reinterpret_cast<cublasHandle_t>(blas_handle);
        if (handle == nullptr || network.params.ptr == nullptr || network.gradients_tmp.ptr == nullptr || input == nullptr || doutput == nullptr || forward_hidden == nullptr || backward_hidden == nullptr) return false;
        if (rows == 0u || rows % kFusedBatchQuantum != 0u) return false;
        constexpr dim3 threads           = {kWmmaThreadsX, kWmmaThreadsY, kWmmaThreadsZ};
        const dim3 blocks                = {rows / kFusedBatchQuantum, 1u, 1u};
        constexpr std::size_t shmem_size = kFusedBackwardShmem;

        k_fully_fused_backward<<<blocks, threads, shmem_size, stream>>>(doutput, network.params.ptr + static_cast<std::uint64_t>(kFusedWidth) * network.input_width, backward_hidden, forward_hidden, rows, network.hidden_matmuls);
        if (cudaGetLastError() != cudaSuccess) return false;

        constexpr float alpha              = kBlasAlpha;
        constexpr float beta               = kBlasBeta;
        std::uint64_t gradient_offset      = 0u;
        const __half* const first_hidden   = forward_hidden;
        const __half* const last_hidden    = forward_hidden + static_cast<std::uint64_t>(network.hidden_matmuls) * kFusedWidth * rows;
        const __half* const first_backprop = backward_hidden;
        const __half* const last_backprop  = backward_hidden + static_cast<std::uint64_t>(network.hidden_matmuls) * kFusedWidth * rows;
        constexpr dim3 input_grad_threads{kInputGradThreadsX, kInputGradThreadsY, kInputGradThreadsZ};

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

    __global__ void k_pack_train_density_input(const nerf::sampler::SampleStep* __restrict__ sample_steps, const std::uint32_t rows, const std::uint32_t padded_rows, __half* __restrict__ density_input) {
        const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const std::uint32_t total = padded_rows * kDensityInputDim;
        if (idx >= total) return;

        const std::uint32_t row = idx / kDensityInputDim;
        const std::uint32_t col = idx - row * kDensityInputDim;
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

    __global__ void k_pack_train_color_input(const nerf::sampler::SampleStep* __restrict__ sample_steps, const __half* __restrict__ density_output, const std::uint32_t rows, const std::uint32_t padded_rows, __half* __restrict__ color_input) {
        const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const std::uint32_t total = padded_rows * kColorInputDim;
        if (idx >= total) return;

        const std::uint32_t row = idx / kColorInputDim;
        const std::uint32_t col = idx - row * kColorInputDim;
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

    __global__ void k_pack_color_output_grad(const float* __restrict__ d_rgb, const std::uint32_t rows, const std::uint32_t padded_rows, const float loss_scale, __half* __restrict__ color_doutput) {
        const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const std::uint32_t total = padded_rows * kColorOutputPaddedDim;
        if (idx >= total) return;

        const std::uint32_t row = idx / kColorOutputPaddedDim;
        const std::uint32_t col = idx - row * kColorOutputPaddedDim;
        float value             = 0.0f;
        if (row < rows && col < kColorOutputDim) value = d_rgb[static_cast<std::uint64_t>(row) * 3ull + col] * loss_scale;
        color_doutput[idx] = __float2half_rn(value);
    }

    __global__ void k_pack_density_output_grad(const float* __restrict__ d_sigma, const __half* __restrict__ color_dinput, const std::uint32_t rows, const std::uint32_t padded_rows, const float loss_scale, __half* __restrict__ density_doutput) {
        const std::uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const std::uint32_t total = padded_rows * kDensityOutputDim;
        if (idx >= total) return;

        const std::uint32_t row = idx / kDensityOutputDim;
        const std::uint32_t col = idx - row * kDensityOutputDim;
        float value             = 0.0f;
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

    bool run_network_inference(NetworkSet& network_set, NetworkWorkspace& workspace, cudaStream_t stream, const NetworkInferenceRequest& request) {
        const std::uint32_t rows          = request.rows;
        const std::uint32_t padded_rows   = (rows + kNetworkBatchGranularity - 1u) / kNetworkBatchGranularity * kNetworkBatchGranularity;
        constexpr std::uint32_t threads   = kThreads256;
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

    __global__ void k_ray_march_mse_grad(const float* __restrict__ raw_rgb, const float* __restrict__ raw_sigma, const nerf::sampler::SampleStep* __restrict__ sample_steps, const nerf::sampler::SampleRay* __restrict__ rays, const nerf::sampler::SampleBatchState* __restrict__ batch_state, std::uint32_t ray_count, float* __restrict__ d_raw_rgb, float* __restrict__ d_raw_sigma, float* __restrict__ trans_tmp, float* __restrict__ loss_sum) {
        const std::uint32_t local_ray   = blockIdx.x * blockDim.x + threadIdx.x;
        float ray_loss                  = 0.0f;
        const std::uint32_t active_rays = batch_state->active_ray_count;
        const float inv_norm            = active_rays == 0u ? 0.0f : 1.0f / static_cast<float>(active_rays);
        if (local_ray < ray_count) {
            const nerf::sampler::SampleRay ray = rays[local_ray];
            const std::uint32_t count          = ray.sample_count;
            if (count > 0u) {
                const std::uint32_t base = ray.sample_begin;

                float T       = 1.0f;
                float accum_r = 0.0f;
                float accum_g = 0.0f;
                float accum_b = 0.0f;

                for (std::uint32_t i = 0; i < count; ++i) {
                    const std::uint32_t s = base + i;
                    const float dt        = fmaxf(0.0f, sample_steps[s].dt);

                    const float sig_raw = raw_sigma[s];
                    const float sigma   = softplus_sigma(sig_raw);
                    const float a       = 1.0f - __expf(-(sigma * dt));

                    const float r = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 0ull]);
                    const float g = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 1ull]);
                    const float b = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 2ull]);

                    trans_tmp[s]  = T;
                    const float w = T * a;

                    accum_r += w * r;
                    accum_g += w * g;
                    accum_b += w * b;

                    T *= 1.0f - a;
                }

                const float pred_r = accum_r + T;
                const float pred_g = accum_g + T;
                const float pred_b = accum_b + T;

                const float gt_r = ray.gt_r;
                const float gt_g = ray.gt_g;
                const float gt_b = ray.gt_b;
                const float gt_a = fminf(1.0f, fmaxf(0.0f, ray.gt_a));

                const float ray_w = 0.1f + 0.9f * gt_a;

                const float dr = pred_r - gt_r;
                const float dg = pred_g - gt_g;
                const float db = pred_b - gt_b;
                ray_loss       = ray_w * (dr * dr + dg * dg + db * db);

                const float g_r = 2.0f * ray_w * dr * inv_norm;
                const float g_g = 2.0f * ray_w * dg * inv_norm;
                const float g_b = 2.0f * ray_w * db * inv_norm;

                float suffix_r = T;
                float suffix_g = T;
                float suffix_b = T;

                for (int i = static_cast<int>(count) - 1; i >= 0; --i) {
                    const std::uint32_t s        = base + static_cast<std::uint32_t>(i);
                    const float dt               = fmaxf(0.0f, sample_steps[s].dt);
                    const float sig_raw          = raw_sigma[s];
                    const float sigma            = softplus_sigma(sig_raw);
                    const float exp_neg_sigma_dt = __expf(-(sigma * dt));
                    const float a                = 1.0f - exp_neg_sigma_dt;
                    const float one_minus_a      = fmaxf(1e-6f, 1.0f - a);

                    const float r       = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 0ull]);
                    const float g       = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 1ull]);
                    const float b       = sigmoid_raw(raw_rgb[static_cast<std::uint64_t>(s) * 3ull + 2ull]);
                    const float trans_i = trans_tmp[s];
                    const float w       = trans_i * a;

                    const float dC_da_r = trans_i * r - suffix_r / one_minus_a;
                    const float dC_da_g = trans_i * g - suffix_g / one_minus_a;
                    const float dC_da_b = trans_i * b - suffix_b / one_minus_a;
                    const float dL_da   = g_r * dC_da_r + g_g * dC_da_g + g_b * dC_da_b;

                    const float dsigma_draw = sigmoid_raw(sig_raw);
                    d_raw_sigma[s]          = dL_da * dt * exp_neg_sigma_dt * dsigma_draw;

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

    bool run_network_training(NetworkSet& network_set, NetworkWorkspace& workspace, cudaStream_t stream, const NetworkTrainingRequest& request) {
        const std::uint32_t padded_rows = (request.sample_count + kNetworkBatchGranularity - 1u) / kNetworkBatchGranularity * kNetworkBatchGranularity;
        constexpr std::uint32_t threads = kThreads256;
        bool ok                         = false;

        do {
            k_pack_train_density_input<<<(padded_rows * kDensityInputDim + threads - 1u) / threads, threads, 0, stream>>>(request.sample_steps, request.sample_count, padded_rows, workspace.density_input);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_forward(network_set.density, stream, workspace.density_input, padded_rows, workspace.density_output, workspace.density_forward_hidden)) break;

            k_pack_train_color_input<<<(padded_rows * kColorInputDim + threads - 1u) / threads, threads, 0, stream>>>(request.sample_steps, workspace.density_output, request.sample_count, padded_rows, workspace.color_input);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_forward(network_set.color, stream, workspace.color_input, padded_rows, workspace.color_output, workspace.color_forward_hidden)) break;

            k_unpack_network_outputs<<<(request.sample_count + threads - 1u) / threads, threads, 0, stream>>>(workspace.density_output, workspace.color_output, request.sample_count, workspace.raw_rgb, workspace.raw_sigma);
            if (cudaGetLastError() != cudaSuccess) break;

            if (cudaMemsetAsync(workspace.d_rgb, 0, static_cast<std::uint64_t>(request.sample_count) * 3ull * sizeof(float), stream) != cudaSuccess) break;
            if (cudaMemsetAsync(workspace.d_sigma, 0, static_cast<std::uint64_t>(request.sample_count) * sizeof(float), stream) != cudaSuccess) break;

            k_ray_march_mse_grad<<<(request.ray_count + threads - 1u) / threads, threads, 0, stream>>>(workspace.raw_rgb, workspace.raw_sigma, request.sample_steps, request.sample_rays, request.batch_state, request.ray_count, workspace.d_rgb, workspace.d_sigma, workspace.trans_tmp, workspace.loss_sum);
            if (cudaGetLastError() != cudaSuccess) break;

            k_pack_color_output_grad<<<(padded_rows * kColorOutputPaddedDim + threads - 1u) / threads, threads, 0, stream>>>(workspace.d_rgb, request.sample_count, padded_rows, kNetworkLossScale, workspace.color_doutput);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_backward(network_set.color, network_set.blas_handle, stream, workspace.color_input, workspace.color_doutput, padded_rows, kGeoFeatureDim, workspace.color_dinput, workspace.color_backward_hidden, workspace.color_forward_hidden)) break;
            k_accumulate_gradients_half<<<(static_cast<std::uint32_t>(network_set.color.gradients.count) + threads - 1u) / threads, threads, 0, stream>>>(network_set.color.gradients.ptr, network_set.color.gradients_tmp.ptr, static_cast<std::uint32_t>(network_set.color.gradients.count));

            k_pack_density_output_grad<<<(padded_rows * kDensityOutputDim + threads - 1u) / threads, threads, 0, stream>>>(workspace.d_sigma, workspace.color_dinput, request.sample_count, padded_rows, kNetworkLossScale, workspace.density_doutput);
            if (cudaGetLastError() != cudaSuccess) break;
            if (!fully_fused_mlp_backward(network_set.density, network_set.blas_handle, stream, workspace.density_input, workspace.density_doutput, padded_rows, 0u, nullptr, workspace.density_backward_hidden, workspace.density_forward_hidden)) break;
            k_accumulate_gradients_half<<<(static_cast<std::uint32_t>(network_set.density.gradients.count) + threads - 1u) / threads, threads, 0, stream>>>(network_set.density.gradients.ptr, network_set.density.gradients_tmp.ptr, static_cast<std::uint32_t>(network_set.density.gradients.count));

            ok = cudaGetLastError() == cudaSuccess;
        } while (false);
        return ok;
    }
} // namespace nerf::network


namespace nerf::runtime {
    struct TrainingDeviceState {
        std::uint32_t frame_index      = 0u;
        std::uint32_t optimizer_step   = 0u;
        std::uint32_t train_camera_idx = 0u;
        NerfTrainStats stats{};
    };

    struct TrainStepConfig {
        float learning_rate;
        float adam_beta1;
        float adam_beta2;
        float adam_eps;
        std::uint32_t lr_decay_ksteps;
    };

    struct OccupancyUpdateRequest {
        const TrainingDeviceState* device_state = nullptr;
        std::uint32_t* bitfield                 = nullptr;
        std::uint64_t bitfield_bytes            = 0u;
        float* density_grid                     = nullptr;
        std::uint32_t grid_res                  = 0u;
        std::uint32_t max_update_tiles          = 0u;
        float decay                             = 0.98f;
        float threshold                         = 0.01f;
        std::uint32_t cells_per_update          = 65536u;
        std::uint32_t update_interval           = 1u;
        std::uint32_t warmup_steps              = 32u;
        float3 aabb_min{};
        float3 aabb_max{};
    };

    struct TrainingStepRequest {
        const nerf::sampler::SampleRay* sample_rays        = nullptr;
        const nerf::sampler::SampleStep* sample_steps      = nullptr;
        const nerf::sampler::SampleBatchState* batch_state = nullptr;
        std::uint32_t camera_count                         = 0u;
        std::uint32_t max_sample_step_count                = 0u;
        TrainStepConfig train_cfg{};
    };

    struct TrainingPlan {
        bool configured = false;
        NerfTrainingConfig config{};
        OccupancyUpdateRequest occupancy_request{};
        nerf::sampler::SamplerRequest sampler_request{};
        TrainingStepRequest training_request{};
    };

    struct TrainRuntime {
        cudaStream_t stream = nullptr;
        nerf::network::NetworkSet network{};
        nerf::network::NetworkWorkspace workspace{};
        TrainingDeviceState* device_state = nullptr;
        std::uint32_t host_frame_index    = 0u;
        TrainingPlan plan{};
        std::mutex run_mutex{};
    };

    struct DeviceContext {
        std::mutex train_runtime_mutex{};
        std::shared_ptr<TrainRuntime> train_runtime{};
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

    __global__ void k_begin_training_step(TrainingDeviceState* state, const std::uint32_t camera_idx) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) return;
        state->train_camera_idx = camera_idx;
    }

    __global__ void k_finalize_training_stats(TrainingDeviceState* state, const nerf::sampler::SampleBatchState* batch_state, const float* loss_sum, const float* grad_sumsq, const std::uint32_t* nonfinite_flag) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) return;
        const std::uint32_t active_rays   = batch_state->active_ray_count;
        const float inv_norm              = active_rays == 0u ? 0.0f : 1.0f / static_cast<float>(active_rays);
        const float loss                  = *loss_sum * inv_norm;
        const float grad_norm             = sqrtf(fmaxf(0.0f, *grad_sumsq)) / kNetworkLossScale;
        const std::uint32_t has_nonfinite = *nonfinite_flag != 0u || batch_state->overflow != 0u || !isfinite(loss) || !isfinite(grad_norm) ? 1u : 0u;
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

    __global__ void k_adam_step_half(float* params_f32, __half* params, const __half* grads, float* adam_m, float* adam_v, const std::uint32_t n, const float base_learning_rate, const float beta1, const float beta2, const float epsilon, const std::uint32_t lr_decay_ksteps, const float inv_loss_scale, const TrainingDeviceState* state) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        const float grad_norm             = state->stats.grad_norm;
        const std::uint32_t has_nonfinite = state->stats.has_nonfinite;
        const bool skip_update            = has_nonfinite != 0u || !isfinite(grad_norm) || grad_norm > kUpdateGuardGradNorm;
        if (skip_update) return;

        const float decay                = static_cast<float>(lr_decay_ksteps) * 1000.0f;
        const float learning_rate        = decay > 0.0f ? base_learning_rate * powf(0.1f, static_cast<float>(state->optimizer_step) / decay) : base_learning_rate;
        const float grad_scale           = grad_norm > kGlobalGradClipNorm && grad_norm > 0.0f ? kGlobalGradClipNorm / grad_norm : 1.0f;
        const float inv_bias_correction1 = 1.0f / (1.0f - powf(beta1, static_cast<float>(state->optimizer_step)));
        const float inv_bias_correction2 = 1.0f / (1.0f - powf(beta2, static_cast<float>(state->optimizer_step)));
        const float grad                 = __half2float(grads[idx]) * inv_loss_scale;
        const float scaled_grad          = grad * grad_scale;
        const float m                    = beta1 * adam_m[idx] + (1.0f - beta1) * scaled_grad;
        const float v                    = beta2 * adam_v[idx] + (1.0f - beta2) * scaled_grad * scaled_grad;
        const float step                 = learning_rate * (m * inv_bias_correction1) / (sqrtf(v * inv_bias_correction2) + epsilon);
        const float w                    = params_f32[idx] - step;

        adam_m[idx]     = m;
        adam_v[idx]     = v;
        params_f32[idx] = w;
        params[idx]     = __float2half_rn(w);
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
        if (rt.network.blas_handle) {
            (void) cublasDestroy(reinterpret_cast<cublasHandle_t>(rt.network.blas_handle));
            rt.network.blas_handle = nullptr;
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

        rt.network          = nerf::network::NetworkSet{};
        rt.plan             = TrainingPlan{};
        rt.host_frame_index = 0u;
    }

    static bool alloc_workspace(nerf::network::NetworkWorkspace& workspace, const std::uint32_t rows) {
        const std::uint32_t padded_rows = (rows + kNetworkBatchGranularity - 1u) / kNetworkBatchGranularity * kNetworkBatchGranularity;
        if (workspace.rows_capacity >= padded_rows) return true;

        free_workspace(workspace);

        const std::uint64_t row_count       = padded_rows;
        constexpr std::uint64_t arena_align = kArenaAlignBytes;
        const auto align_up                 = [](const std::uint64_t value, const std::uint64_t alignment) -> std::uint64_t { return value + alignment - 1u & ~(alignment - 1u); };

        std::uint64_t total = 0u;
        const auto reserve  = [&](const std::uint64_t bytes) {
            total = align_up(total, arena_align);
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
            offset = align_up(offset, arena_align);
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
            rt.network.density.input_width      = kDensityInputDim;
            rt.network.density.width            = kDensityWidth;
            rt.network.density.output_width     = kDensityOutputDim;
            rt.network.density.hidden_matmuls   = kDensityHiddenLayers - 1u;
            rt.network.density.params_f32.count = static_cast<std::uint64_t>(kDensityWidth) * kDensityInputDim + static_cast<std::uint64_t>(rt.network.density.hidden_matmuls) * kDensityWidth * kDensityWidth + static_cast<std::uint64_t>(kDensityOutputDim) * kDensityWidth;
            rt.network.color.input_width        = kColorInputDim;
            rt.network.color.width              = kColorWidth;
            rt.network.color.output_width       = kColorOutputPaddedDim;
            rt.network.color.hidden_matmuls     = kColorHiddenLayers - 1u;
            rt.network.color.params_f32.count   = static_cast<std::uint64_t>(kColorWidth) * kColorInputDim + static_cast<std::uint64_t>(rt.network.color.hidden_matmuls) * kColorWidth * kColorWidth + static_cast<std::uint64_t>(kColorOutputPaddedDim) * kColorWidth;

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

            constexpr std::uint32_t convert_threads = kConvertThreads;
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

            if (!alloc_workspace(rt.workspace, kTrainChunkRows + kNetworkBatchGranularity)) return false;
            if (!nerf::network::init_network_module(rt.network, rt.stream)) return false;
            if (cudaStreamSynchronize(rt.stream) != cudaSuccess) return false;
            return true;
        } catch (...) {
            destroy_runtime(rt);
            return false;
        }
    }

    static std::shared_ptr<TrainRuntime> runtime_for_context(DeviceContext* device_context) {
        std::scoped_lock lock(device_context->train_runtime_mutex);
        if (device_context->train_runtime) return device_context->train_runtime;

        const auto runtime = std::make_shared<TrainRuntime>();
        if (!init_runtime(*runtime)) return nullptr;

        device_context->train_runtime = runtime;
        return device_context->train_runtime;
    }

    static bool destroy_runtime_for_context(DeviceContext* device_context) {
        std::shared_ptr<TrainRuntime> runtime;
        {
            std::scoped_lock lock(device_context->train_runtime_mutex);
            if (!device_context->train_runtime) return true;
            runtime = std::move(device_context->train_runtime);
        }

        std::scoped_lock run_lock(runtime->run_mutex);
        destroy_runtime(*runtime);
        return true;
    }

    static bool prepare_training_device_state(TrainRuntime& runtime, const std::uint32_t camera_idx) {
        k_begin_training_step<<<1, 1, 0, runtime.stream>>>(runtime.device_state, camera_idx);
        return cudaGetLastError() == cudaSuccess;
    }

    static bool read_training_device_stats(DeviceContext* device_context, NerfTrainStats* out_stats) {
        if (!out_stats) return false;
        const auto runtime = runtime_for_context(device_context);
        if (!runtime) return false;
        std::scoped_lock run_lock(runtime->run_mutex);
        if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return false;
        if (cudaMemcpy(out_stats, &runtime->device_state->stats, sizeof(NerfTrainStats), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        runtime->host_frame_index = out_stats->completed_steps;
        return true;
    }

    static bool runtime_zero_network_gradients(TrainRuntime& runtime) {
        if (cudaMemsetAsync(runtime.network.density.gradients.ptr, 0, runtime.network.density.gradients.bytes, runtime.stream) != cudaSuccess) return false;
        if (cudaMemsetAsync(runtime.network.color.gradients.ptr, 0, runtime.network.color.gradients.bytes, runtime.stream) != cudaSuccess) return false;
        return true;
    }

    static bool runtime_zero_scalar_buffers(TrainRuntime& runtime) {
        if (cudaMemsetAsync(runtime.workspace.loss_sum, 0, sizeof(float), runtime.stream) != cudaSuccess) return false;
        if (cudaMemsetAsync(runtime.workspace.grad_sumsq, 0, sizeof(float), runtime.stream) != cudaSuccess) return false;
        if (cudaMemsetAsync(runtime.workspace.nonfinite_flag, 0, sizeof(std::uint32_t), runtime.stream) != cudaSuccess) return false;
        return true;
    }

    static void runtime_accumulate_grad_stats(TrainRuntime& runtime) {
        constexpr std::uint32_t threads = kThreads256;
        const std::uint32_t density_n   = static_cast<std::uint32_t>(runtime.network.density.gradients.count);
        const std::uint32_t color_n     = static_cast<std::uint32_t>(runtime.network.color.gradients.count);

        if (density_n != 0u) {
            k_accum_grad_stats_half<<<(density_n + threads - 1u) / threads, threads, 0, runtime.stream>>>(runtime.network.density.gradients.ptr, density_n, runtime.workspace.grad_sumsq, runtime.workspace.nonfinite_flag);
        }
        if (color_n != 0u) {
            k_accum_grad_stats_half<<<(color_n + threads - 1u) / threads, threads, 0, runtime.stream>>>(runtime.network.color.gradients.ptr, color_n, runtime.workspace.grad_sumsq, runtime.workspace.nonfinite_flag);
        }
    }

    static bool runtime_apply_optimizer(TrainRuntime& runtime, const TrainStepConfig& config) {
        const float beta1   = config.adam_beta1;
        const float beta2   = config.adam_beta2;
        const float epsilon = config.adam_eps;

        if (!(beta1 >= 0.0f && beta1 < 1.0f && beta2 >= 0.0f && beta2 < 1.0f && epsilon > 0.0f && std::isfinite(epsilon))) return false;

        constexpr std::uint32_t threads = kThreads256;
        constexpr float inv_loss_scale  = kInvLossScale;
        const std::uint32_t density_n   = static_cast<std::uint32_t>(runtime.network.density.gradients.count);
        const std::uint32_t color_n     = static_cast<std::uint32_t>(runtime.network.color.gradients.count);
        if (density_n != 0u) {
            k_adam_step_half<<<(density_n + threads - 1u) / threads, threads, 0, runtime.stream>>>(runtime.network.density.params_f32.ptr, runtime.network.density.params.ptr, runtime.network.density.gradients.ptr, runtime.network.density.adam_m.ptr, runtime.network.density.adam_v.ptr, density_n, config.learning_rate, beta1, beta2, epsilon, config.lr_decay_ksteps, inv_loss_scale, runtime.device_state);
        }
        if (color_n != 0u) {
            k_adam_step_half<<<(color_n + threads - 1u) / threads, threads, 0, runtime.stream>>>(runtime.network.color.params_f32.ptr, runtime.network.color.params.ptr, runtime.network.color.gradients.ptr, runtime.network.color.adam_m.ptr, runtime.network.color.adam_v.ptr, color_n, config.learning_rate, beta1, beta2, epsilon, config.lr_decay_ksteps, inv_loss_scale, runtime.device_state);
        }
        if (cudaGetLastError() != cudaSuccess) return false;
        return true;
    }

    __global__ void k_prepare_update_density_grid(float* data, const std::uint32_t count, const float decay, const std::uint32_t update_interval, const std::uint32_t warmup_steps, const TrainingDeviceState* state) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;
        if (state->frame_index < warmup_steps)
            data[idx] = 1.0f;
        else if (state->frame_index == 0u || state->frame_index % update_interval == 0u)
            data[idx] *= decay;
    }

    __global__ void k_prepare_occupancy_bitfield(std::uint32_t* bitfield, const std::uint32_t word_count, const std::uint32_t update_interval, const std::uint32_t warmup_steps, const TrainingDeviceState* state) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= word_count) return;
        if (state->frame_index < warmup_steps)
            bitfield[idx] = 0xFFFFFFFFu;
        else if (state->frame_index == 0u || state->frame_index % update_interval == 0u)
            bitfield[idx] = 0u;
    }

    __global__ void k_build_occ_inputs(const TrainingDeviceState* state, const std::uint32_t total_rows, const std::uint32_t cells_per_update, const std::uint32_t cell_count, const std::uint32_t grid_res, const std::uint32_t warmup_steps, const std::uint32_t update_interval, const float3 aabb_min, const float3 aabb_max, float* out_inputs) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_rows) return;

        if (state->frame_index < warmup_steps || (state->frame_index >= warmup_steps && state->frame_index != 0u && state->frame_index % update_interval != 0u)) {
            float* dst = out_inputs + static_cast<std::uint64_t>(idx) * 7ull;
            dst[0]     = 0.0f;
            dst[1]     = 0.0f;
            dst[2]     = 0.0f;
            dst[3]     = 0.0f;
            dst[4]     = 0.0f;
            dst[5]     = 0.0f;
            dst[6]     = 1.0f;
            return;
        }
        const std::uint32_t update_count = min(cells_per_update, cell_count);
        const std::uint32_t linear_idx   = idx;
        float* dst                       = out_inputs + static_cast<std::uint64_t>(idx) * 7ull;
        if (linear_idx >= update_count) {
            dst[0] = 0.0f;
            dst[1] = 0.0f;
            dst[2] = 0.0f;
            dst[3] = 0.0f;
            dst[4] = 0.0f;
            dst[5] = 0.0f;
            dst[6] = 1.0f;
            return;
        }

        const std::uint32_t start_cell_base = static_cast<std::uint32_t>(static_cast<std::uint64_t>(state->frame_index) * update_count % cell_count);
        const std::uint32_t cell            = (start_cell_base + linear_idx) % cell_count;
        const std::uint32_t layer           = grid_res * grid_res;
        const std::uint32_t z               = cell / layer;
        const std::uint32_t rem             = cell - z * layer;
        const std::uint32_t y               = rem / grid_res;
        const std::uint32_t x               = rem - y * grid_res;
        const float3 step                   = make_float3((aabb_max.x - aabb_min.x) / static_cast<float>(grid_res), (aabb_max.y - aabb_min.y) / static_cast<float>(grid_res), (aabb_max.z - aabb_min.z) / static_cast<float>(grid_res));
        const float3 p                      = make_float3(aabb_min.x + (static_cast<float>(x) + 0.5f) * step.x, aabb_min.y + (static_cast<float>(y) + 0.5f) * step.y, aabb_min.z + (static_cast<float>(z) + 0.5f) * step.z);

        dst[0] = p.x;
        dst[1] = p.y;
        dst[2] = p.z;
        dst[3] = 0.0f;
        dst[4] = 0.0f;
        dst[5] = 0.0f;
        dst[6] = 1.0f;
    }

    __global__ void k_update_density_from_sigma(float* density_grid, const float* raw_sigma, const TrainingDeviceState* state, const std::uint32_t total_rows, const std::uint32_t cells_per_update, const std::uint32_t cell_count, const std::uint32_t warmup_steps, const std::uint32_t update_interval) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_rows) return;
        if (state->frame_index < warmup_steps) return;
        if (state->frame_index >= warmup_steps && state->frame_index != 0u && state->frame_index % update_interval != 0u) return;
        const std::uint32_t update_count = min(cells_per_update, cell_count);
        const std::uint32_t linear_idx   = idx;
        if (linear_idx >= update_count) return;
        const std::uint32_t start_cell_base = static_cast<std::uint32_t>(static_cast<std::uint64_t>(state->frame_index) * update_count % cell_count);
        const std::uint32_t cell            = (start_cell_base + linear_idx) % cell_count;
        const float sig_raw                 = raw_sigma[idx];
        const float sigma                   = sig_raw > 20.0f ? sig_raw : sig_raw < -20.0f ? __expf(sig_raw) : log1pf(__expf(sig_raw));
        density_grid[cell]                  = fmaxf(density_grid[cell], sigma);
    }

    __global__ void k_rebuild_occ_from_density(const float* density_grid, const std::uint32_t cell_count, const float threshold, const std::uint32_t warmup_steps, const std::uint32_t update_interval, const TrainingDeviceState* state, std::uint32_t* bitfield) {
        const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= cell_count) return;
        if (state->frame_index < warmup_steps) return;
        if (state->frame_index >= warmup_steps && state->frame_index != 0u && state->frame_index % update_interval != 0u) return;
        if (density_grid[idx] <= threshold) return;
        atomicOr(&bitfield[idx >> 5u], 1u << (idx & 31u));
    }

    static bool runtime_run_occupancy_update(TrainRuntime& runtime, const OccupancyUpdateRequest& request) {
        const std::uint32_t cell_count  = static_cast<std::uint32_t>(static_cast<std::uint64_t>(request.grid_res) * static_cast<std::uint64_t>(request.grid_res) * static_cast<std::uint64_t>(request.grid_res));
        const std::uint32_t word_count  = static_cast<std::uint32_t>((request.bitfield_bytes + sizeof(std::uint32_t) - 1u) / sizeof(std::uint32_t));
        const std::uint32_t total_rows  = request.max_update_tiles * kTrainChunkRows;
        constexpr std::uint32_t block_x = kOccupancyBlockX;
        const dim3 full_grid((cell_count + block_x - 1u) / block_x);
        const dim3 word_grid((word_count + block_x - 1u) / block_x);
        const dim3 tile_grid((total_rows + block_x - 1u) / block_x);

        k_prepare_update_density_grid<<<full_grid, block_x, 0, runtime.stream>>>(request.density_grid, cell_count, request.decay, request.update_interval, request.warmup_steps, request.device_state);
        if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) return false;
        k_prepare_occupancy_bitfield<<<word_grid, block_x, 0, runtime.stream>>>(request.bitfield, word_count, request.update_interval, request.warmup_steps, request.device_state);
        if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) return false;
        if (total_rows == 0u) return true;

        k_build_occ_inputs<<<tile_grid, block_x, 0, runtime.stream>>>(request.device_state, total_rows, request.cells_per_update, cell_count, request.grid_res, request.warmup_steps, request.update_interval, request.aabb_min, request.aabb_max, runtime.workspace.inputs_tmp);
        if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) return false;
        if (!nerf::encoder::run_encoder_module(runtime.stream, runtime.workspace.inputs_tmp, total_rows, runtime.workspace.enc_pts, runtime.workspace.enc_dir)) return false;
        if (!nerf::network::run_network_inference(runtime.network, runtime.workspace, runtime.stream, nerf::network::NetworkInferenceRequest{.encoded_pts = runtime.workspace.enc_pts, .encoded_dir = runtime.workspace.enc_dir, .rows = total_rows, .raw_rgb = runtime.workspace.raw_rgb, .raw_sigma = runtime.workspace.raw_sigma})) return false;
        k_update_density_from_sigma<<<tile_grid, block_x, 0, runtime.stream>>>(request.density_grid, runtime.workspace.raw_sigma, request.device_state, total_rows, request.cells_per_update, cell_count, request.warmup_steps, request.update_interval);
        if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) return false;
        k_rebuild_occ_from_density<<<full_grid, block_x, 0, runtime.stream>>>(request.density_grid, cell_count, request.threshold, request.warmup_steps, request.update_interval, request.device_state, request.bitfield);
        return cudaGetLastError() == cudaSuccess;
    }

    static bool runtime_run_training_step(TrainRuntime& runtime, const TrainingStepRequest& request, const nerf::sampler::SampleBatchState& host_batch_state) {
        if (!runtime_zero_network_gradients(runtime)) return false;
        if (!runtime_zero_scalar_buffers(runtime)) return false;

        nerf::network::NetworkTrainingRequest training_request{};
        training_request.sample_rays  = request.sample_rays;
        training_request.sample_steps = request.sample_steps;
        training_request.batch_state  = request.batch_state;
        training_request.ray_count    = host_batch_state.active_ray_count;
        training_request.sample_count = host_batch_state.sample_step_count;
        if (training_request.ray_count != 0u && training_request.sample_count != 0u) {
            if (!nerf::network::run_network_training(runtime.network, runtime.workspace, runtime.stream, training_request)) return false;
        }

        runtime_accumulate_grad_stats(runtime);
        if (cudaGetLastError() != cudaSuccess) return false;
        k_finalize_training_stats<<<1, 1, 0, runtime.stream>>>(runtime.device_state, request.batch_state, runtime.workspace.loss_sum, runtime.workspace.grad_sumsq, runtime.workspace.nonfinite_flag);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (!runtime_apply_optimizer(runtime, request.train_cfg)) return false;
        k_commit_training_step<<<1, 1, 0, runtime.stream>>>(runtime.device_state);
        return cudaGetLastError() == cudaSuccess;
    }
    static bool configure_training_plan(TrainRuntime& runtime, const OccupancyUpdateRequest& occupancy_request, const nerf::sampler::SamplerRequest& sampler_request, const TrainingStepRequest& training_request, const NerfTrainingConfig& config) {
        std::scoped_lock run_lock(runtime.run_mutex);

        const bool same_plan = runtime.plan.configured && runtime.plan.training_request.max_sample_step_count == training_request.max_sample_step_count && runtime.plan.occupancy_request.max_update_tiles == occupancy_request.max_update_tiles && runtime.plan.config.aabb_min.x == config.aabb_min.x && runtime.plan.config.aabb_min.y == config.aabb_min.y && runtime.plan.config.aabb_min.z == config.aabb_min.z && runtime.plan.config.aabb_max.x == config.aabb_max.x
                            && runtime.plan.config.aabb_max.y == config.aabb_max.y && runtime.plan.config.aabb_max.z == config.aabb_max.z && runtime.plan.config.hyper_params.learning_rate == config.hyper_params.learning_rate && runtime.plan.config.hyper_params.adam_beta1 == config.hyper_params.adam_beta1 && runtime.plan.config.hyper_params.adam_beta2 == config.hyper_params.adam_beta2 && runtime.plan.config.hyper_params.adam_eps == config.hyper_params.adam_eps
                            && runtime.plan.config.hyper_params.lr_decay_ksteps == config.hyper_params.lr_decay_ksteps && runtime.plan.config.occupancy_params.decay == config.occupancy_params.decay && runtime.plan.config.occupancy_params.threshold == config.occupancy_params.threshold && runtime.plan.config.occupancy_params.cells_per_update == config.occupancy_params.cells_per_update
                            && runtime.plan.config.occupancy_params.update_interval == config.occupancy_params.update_interval && runtime.plan.config.occupancy_params.warmup_steps == config.occupancy_params.warmup_steps && runtime.plan.config.rays_per_batch == config.rays_per_batch && runtime.plan.config.max_sample_steps_per_ray == config.max_sample_steps_per_ray && runtime.plan.config.train_camera_mode == config.train_camera_mode
                            && runtime.plan.config.fixed_train_camera_idx == config.fixed_train_camera_idx;
        if (same_plan) return true;

        if (cudaStreamSynchronize(runtime.stream) != cudaSuccess) return false;
        const std::uint32_t occupancy_rows = occupancy_request.max_update_tiles * kTrainChunkRows;
        const std::uint32_t workspace_rows = std::max(training_request.max_sample_step_count + kNetworkBatchGranularity, occupancy_rows);
        if (!alloc_workspace(runtime.workspace, workspace_rows)) return false;
        runtime.plan                   = TrainingPlan{};
        runtime.plan.config            = config;
        runtime.plan.occupancy_request = occupancy_request;
        runtime.plan.sampler_request   = sampler_request;
        runtime.plan.training_request  = training_request;
        runtime.plan.configured        = true;
        return true;
    }

    static bool run_training_plan(DeviceContext* device_context) {
        const auto runtime = runtime_for_context(device_context);
        if (!runtime) return false;
        std::scoped_lock run_lock(runtime->run_mutex);
        if (!runtime->plan.configured) return false;

        const std::uint32_t frame_index  = runtime->host_frame_index;
        const std::uint32_t camera_count = runtime->plan.training_request.camera_count;
        std::uint32_t camera_idx         = 0u;
        if (camera_count != 0u) {
            switch (runtime->plan.config.train_camera_mode) {
            case NERF_TRAIN_CAMERA_MODE_FIXED: camera_idx = runtime->plan.config.fixed_train_camera_idx % camera_count; break;
            case NERF_TRAIN_CAMERA_MODE_CYCLE: camera_idx = frame_index % camera_count; break;
            case NERF_TRAIN_CAMERA_MODE_RANDOM: camera_idx = hash_u32(frame_index * 747796405u + 2891336453u) % camera_count; break;
            default: return false;
            }
        }

        nerf::sampler::SamplerRequest sampler_request = runtime->plan.sampler_request;
        sampler_request.frame_index                   = frame_index;
        sampler_request.camera_idx                    = camera_idx;

        if (!runtime_run_occupancy_update(*runtime, runtime->plan.occupancy_request)) return false;
        if (!prepare_training_device_state(*runtime, camera_idx)) return false;
        if (!nerf::sampler::run_sampler(sampler_request)) return false;
        nerf::sampler::SampleBatchState host_batch_state{};
        if (cudaMemcpyAsync(&host_batch_state, sampler_request.batch_state, sizeof(nerf::sampler::SampleBatchState), cudaMemcpyDeviceToHost, runtime->stream) != cudaSuccess) return false;
        if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return false;
        if (host_batch_state.overflow != 0u) {
            if (!runtime_zero_network_gradients(*runtime)) return false;
            if (!runtime_zero_scalar_buffers(*runtime)) return false;
            k_finalize_training_stats<<<1, 1, 0, runtime->stream>>>(runtime->device_state, runtime->plan.training_request.batch_state, runtime->workspace.loss_sum, runtime->workspace.grad_sumsq, runtime->workspace.nonfinite_flag);
            if (cudaGetLastError() != cudaSuccess) return false;
            if (!runtime_apply_optimizer(*runtime, runtime->plan.training_request.train_cfg)) return false;
            k_commit_training_step<<<1, 1, 0, runtime->stream>>>(runtime->device_state);
            if (cudaGetLastError() != cudaSuccess) return false;
            ++runtime->host_frame_index;
            return true;
        }

        if (!runtime_run_training_step(*runtime, runtime->plan.training_request, host_batch_state)) return false;
        ++runtime->host_frame_index;
        return true;
    }
} // namespace nerf::runtime

namespace {
    struct Region {
        std::uint64_t offset_bytes = 0u;
        std::uint64_t size_bytes   = 0u;
    };

    struct DeviceSpan {
        std::byte* ptr           = nullptr;
        std::uint64_t size_bytes = 0u;
    };

    struct ContextStorage {
        nerf::runtime::DeviceContext* cuda_context = nullptr;
        std::byte* scratch_device_base             = nullptr;
        std::byte* scene_device_base               = nullptr;
        DeviceSpan images{};
        DeviceSpan xforms{};
        DeviceSpan occupancy_bitfield{};
        DeviceSpan occupancy_density{};
        DeviceSpan sample_rays{};
        DeviceSpan sample_steps{};
        DeviceSpan sample_batch_state{};
        NerfDatasetInfo dataset_info{};
        std::uint32_t occupancy_grid_res    = 0u;
        std::uint32_t max_sample_steps      = 0u;
        std::uint32_t max_batch_rays        = 0u;
        std::uint64_t arena_alignment_bytes = 0u;
        bool dataset_loaded                 = false;
        bool training_configured            = false;
        NerfTrainingConfig training_config{};
    };
} // namespace

extern "C" {
NerfStatus nerf_create_context(const NerfCreateDesc* desc, void** out_context) {
    if (!desc || !out_context) return NERF_STATUS_INVALID_ARGUMENT;

    const NerfCreateDesc normalized = *desc;
    if (normalized.occupancy_grid_res == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.max_sample_steps == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.max_batch_rays == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.arena_alignment_bytes == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if ((normalized.arena_alignment_bytes & normalized.arena_alignment_bytes - 1u) != 0u) return NERF_STATUS_INVALID_ARGUMENT;

    std::uint64_t cursor = 0u;
    Region occupancy_bitfield{};
    Region occupancy_density{};
    {
        std::uint64_t cell_count = normalized.occupancy_grid_res;
        if (cell_count > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(normalized.occupancy_grid_res)) return NERF_STATUS_OVERFLOW;
        cell_count *= static_cast<std::uint64_t>(normalized.occupancy_grid_res);
        if (cell_count > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(normalized.occupancy_grid_res)) return NERF_STATUS_OVERFLOW;
        cell_count *= static_cast<std::uint64_t>(normalized.occupancy_grid_res);

        const std::uint64_t bitfield_bytes = (cell_count + 31u) / 32u * sizeof(std::uint32_t);
        std::uint64_t density_bytes        = cell_count;
        if (density_bytes > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) return NERF_STATUS_OVERFLOW;
        density_bytes *= sizeof(float);

        if (normalized.arena_alignment_bytes > 1u) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - (normalized.arena_alignment_bytes - 1u)) return NERF_STATUS_OVERFLOW;
            cursor = cursor + normalized.arena_alignment_bytes - 1u & ~(normalized.arena_alignment_bytes - 1u);
        }
        occupancy_bitfield.offset_bytes = cursor;
        occupancy_bitfield.size_bytes   = bitfield_bytes;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - bitfield_bytes) return NERF_STATUS_OVERFLOW;
        cursor += bitfield_bytes;

        if (normalized.arena_alignment_bytes > 1u) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - (normalized.arena_alignment_bytes - 1u)) return NERF_STATUS_OVERFLOW;
            cursor = cursor + normalized.arena_alignment_bytes - 1u & ~(normalized.arena_alignment_bytes - 1u);
        }
        occupancy_density.offset_bytes = cursor;
        occupancy_density.size_bytes   = density_bytes;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - density_bytes) return NERF_STATUS_OVERFLOW;
        cursor += density_bytes;
    }

    Region sample_rays{};
    Region sample_steps{};
    Region sample_batch_state{};
    {
        std::uint64_t sample_ray_bytes = normalized.max_batch_rays;
        if (sample_ray_bytes > std::numeric_limits<std::uint64_t>::max() / sizeof(nerf::sampler::SampleRay)) return NERF_STATUS_OVERFLOW;
        sample_ray_bytes *= sizeof(nerf::sampler::SampleRay);

        std::uint64_t sample_step_bytes = normalized.max_sample_steps;
        if (sample_step_bytes > std::numeric_limits<std::uint64_t>::max() / sizeof(nerf::sampler::SampleStep)) return NERF_STATUS_OVERFLOW;
        sample_step_bytes *= sizeof(nerf::sampler::SampleStep);

        if (normalized.arena_alignment_bytes > 1u) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - (normalized.arena_alignment_bytes - 1u)) return NERF_STATUS_OVERFLOW;
            cursor = cursor + normalized.arena_alignment_bytes - 1u & ~(normalized.arena_alignment_bytes - 1u);
        }
        sample_rays.offset_bytes = cursor;
        sample_rays.size_bytes   = sample_ray_bytes;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - sample_ray_bytes) return NERF_STATUS_OVERFLOW;
        cursor += sample_ray_bytes;

        if (normalized.arena_alignment_bytes > 1u) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - (normalized.arena_alignment_bytes - 1u)) return NERF_STATUS_OVERFLOW;
            cursor = cursor + normalized.arena_alignment_bytes - 1u & ~(normalized.arena_alignment_bytes - 1u);
        }
        sample_steps.offset_bytes = cursor;
        sample_steps.size_bytes   = sample_step_bytes;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - sample_step_bytes) return NERF_STATUS_OVERFLOW;
        cursor += sample_step_bytes;

        if (normalized.arena_alignment_bytes > 1u) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - (normalized.arena_alignment_bytes - 1u)) return NERF_STATUS_OVERFLOW;
            cursor = cursor + normalized.arena_alignment_bytes - 1u & ~(normalized.arena_alignment_bytes - 1u);
        }
        sample_batch_state.offset_bytes = cursor;
        sample_batch_state.size_bytes   = sizeof(nerf::sampler::SampleBatchState);
        if (cursor > std::numeric_limits<std::uint64_t>::max() - sample_batch_state.size_bytes) return NERF_STATUS_OVERFLOW;
        cursor += sample_batch_state.size_bytes;
    }

    std::unique_ptr<nerf::runtime::DeviceContext> cuda_context = std::make_unique<nerf::runtime::DeviceContext>();
    std::unique_ptr<ContextStorage> context                    = std::make_unique<ContextStorage>();
    context->cuda_context                                      = cuda_context.release();
    context->occupancy_grid_res                                = normalized.occupancy_grid_res;
    context->max_sample_steps                                  = normalized.max_sample_steps;
    context->max_batch_rays                                    = normalized.max_batch_rays;
    context->arena_alignment_bytes                             = normalized.arena_alignment_bytes;

    if (cursor != 0u) {
        void* scratch_ptr              = nullptr;
        const cudaError_t alloc_status = cudaMalloc(&scratch_ptr, cursor);
        if (alloc_status != cudaSuccess) {
            delete context->cuda_context;
            if (alloc_status == cudaErrorMemoryAllocation) return NERF_STATUS_OUT_OF_MEMORY;
            return NERF_STATUS_CUDA_FAILURE;
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

    if (!nerf::runtime::runtime_for_context(context->cuda_context)) {
        if (context->scratch_device_base) cudaFree(context->scratch_device_base);
        delete context->cuda_context;
        return NERF_STATUS_INTERNAL_ERROR;
    }

    *out_context = context.release();
    return NERF_STATUS_OK;
}

NerfStatus nerf_destroy_context(void* context) {
    if (!context) return NERF_STATUS_OK;

    const std::unique_ptr<ContextStorage> owned{static_cast<ContextStorage*>(context)};
    NerfStatus status = NERF_STATUS_OK;

    if (owned->cuda_context) {
        if (!nerf::runtime::destroy_runtime_for_context(owned->cuda_context)) status = NERF_STATUS_INTERNAL_ERROR;
        delete owned->cuda_context;
        owned->cuda_context = nullptr;
    }
    if (owned->scene_device_base) {
        if (cudaFree(owned->scene_device_base) != cudaSuccess && status == NERF_STATUS_OK) status = NERF_STATUS_CUDA_FAILURE;
        owned->scene_device_base = nullptr;
    }
    if (owned->scratch_device_base) {
        if (cudaFree(owned->scratch_device_base) != cudaSuccess && status == NERF_STATUS_OK) status = NERF_STATUS_CUDA_FAILURE;
        owned->scratch_device_base = nullptr;
    }
    return status;
}

NerfStatus nerf_load_dataset(void* context, const NerfDatasetLoadDesc* desc, NerfDatasetInfo* out_info) {
    if (!context || !desc) return NERF_STATUS_INVALID_ARGUMENT;

    nerf::host::HostDatasetData host_data{};
    const NerfStatus load_status = nerf::io::load_dataset_host_data(*desc, host_data);
    if (load_status != NERF_STATUS_OK) return load_status;

    ContextStorage* ctx = static_cast<ContextStorage*>(context);
    if (!nerf::runtime::destroy_runtime_for_context(ctx->cuda_context)) return NERF_STATUS_INTERNAL_ERROR;

    if (ctx->scene_device_base) {
        if (cudaFree(ctx->scene_device_base) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
        ctx->scene_device_base = nullptr;
    }

    std::uint64_t cursor = 0u;
    Region images{};
    Region xforms{};
    {
        if (ctx->arena_alignment_bytes > 1u) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - (ctx->arena_alignment_bytes - 1u)) return NERF_STATUS_OVERFLOW;
            cursor = cursor + ctx->arena_alignment_bytes - 1u & ~(ctx->arena_alignment_bytes - 1u);
        }
        images.offset_bytes = cursor;
        images.size_bytes   = host_data.info.images_bytes;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - images.size_bytes) return NERF_STATUS_OVERFLOW;
        cursor += images.size_bytes;

        if (ctx->arena_alignment_bytes > 1u) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - (ctx->arena_alignment_bytes - 1u)) return NERF_STATUS_OVERFLOW;
            cursor = cursor + ctx->arena_alignment_bytes - 1u & ~(ctx->arena_alignment_bytes - 1u);
        }
        xforms.offset_bytes = cursor;
        xforms.size_bytes   = host_data.info.c2w_bytes;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - xforms.size_bytes) return NERF_STATUS_OVERFLOW;
        cursor += xforms.size_bytes;
    }

    void* scene_ptr                = nullptr;
    const cudaError_t alloc_status = cudaMalloc(&scene_ptr, cursor);
    if (alloc_status != cudaSuccess) {
        if (alloc_status == cudaErrorMemoryAllocation) return NERF_STATUS_OUT_OF_MEMORY;
        return NERF_STATUS_CUDA_FAILURE;
    }

    ctx->scene_device_base = static_cast<std::byte*>(scene_ptr);
    ctx->images.ptr        = ctx->scene_device_base + images.offset_bytes;
    ctx->images.size_bytes = images.size_bytes;
    ctx->xforms.ptr        = ctx->scene_device_base + xforms.offset_bytes;
    ctx->xforms.size_bytes = xforms.size_bytes;

    if (cudaMemcpy(ctx->images.ptr, host_data.images_rgba8.data(), ctx->images.size_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(ctx->scene_device_base);
        ctx->scene_device_base = nullptr;
        ctx->images            = {};
        ctx->xforms            = {};
        return NERF_STATUS_CUDA_FAILURE;
    }
    if (cudaMemcpy(ctx->xforms.ptr, host_data.c2w_4x4.data(), ctx->xforms.size_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(ctx->scene_device_base);
        ctx->scene_device_base = nullptr;
        ctx->images            = {};
        ctx->xforms            = {};
        return NERF_STATUS_CUDA_FAILURE;
    }

    ctx->dataset_info        = host_data.info;
    ctx->dataset_loaded      = true;
    ctx->training_configured = false;
    ctx->training_config     = NerfTrainingConfig{};

    if (out_info) *out_info = ctx->dataset_info;
    return NERF_STATUS_OK;
}

NerfStatus nerf_configure_training(void* context, const NerfTrainingConfig* config) {
    if (!context || !config) return NERF_STATUS_INVALID_ARGUMENT;

    const NerfTrainingConfig normalized = *config;
    if (!(normalized.aabb_min.x < normalized.aabb_max.x && normalized.aabb_min.y < normalized.aabb_max.y && normalized.aabb_min.z < normalized.aabb_max.z)) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.occupancy_params.cells_per_update == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.occupancy_params.update_interval == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.rays_per_batch == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.max_sample_steps_per_ray == 0u) return NERF_STATUS_INVALID_ARGUMENT;
    if (normalized.max_sample_steps_per_ray > kMaxSampleStepsPerRay) return NERF_STATUS_RANGE_ERROR;
    switch (normalized.train_camera_mode) {
    case NERF_TRAIN_CAMERA_MODE_FIXED:
    case NERF_TRAIN_CAMERA_MODE_CYCLE:
    case NERF_TRAIN_CAMERA_MODE_RANDOM: break;
    default: return NERF_STATUS_INVALID_ARGUMENT;
    }

    ContextStorage* ctx = static_cast<ContextStorage*>(context);
    if (!ctx->dataset_loaded) return NERF_STATUS_DATASET_NOT_LOADED;
    if (normalized.rays_per_batch > ctx->max_batch_rays) return NERF_STATUS_RANGE_ERROR;
    if (normalized.train_camera_mode == NERF_TRAIN_CAMERA_MODE_FIXED && normalized.fixed_train_camera_idx >= ctx->dataset_info.image_count) return NERF_STATUS_RANGE_ERROR;
    const std::uint64_t total_sample_steps = static_cast<std::uint64_t>(normalized.rays_per_batch) * static_cast<std::uint64_t>(normalized.max_sample_steps_per_ray);
    if (total_sample_steps > static_cast<std::uint64_t>(ctx->max_sample_steps)) return NERF_STATUS_RANGE_ERROR;

    const auto runtime = nerf::runtime::runtime_for_context(ctx->cuda_context);
    if (!runtime) return NERF_STATUS_INTERNAL_ERROR;

    const std::uint32_t occupancy_cell_count = static_cast<std::uint32_t>(static_cast<std::uint64_t>(ctx->occupancy_grid_res) * static_cast<std::uint64_t>(ctx->occupancy_grid_res) * static_cast<std::uint64_t>(ctx->occupancy_grid_res));
    const nerf::runtime::OccupancyUpdateRequest occupancy_request{
        .device_state     = runtime->device_state,
        .bitfield         = reinterpret_cast<std::uint32_t*>(ctx->occupancy_bitfield.ptr),
        .bitfield_bytes   = ctx->occupancy_bitfield.size_bytes,
        .density_grid     = reinterpret_cast<float*>(ctx->occupancy_density.ptr),
        .grid_res         = ctx->occupancy_grid_res,
        .max_update_tiles = ((std::min<std::uint32_t>(normalized.occupancy_params.cells_per_update, occupancy_cell_count) + kTrainChunkRows - 1u) / kTrainChunkRows),
        .decay            = normalized.occupancy_params.decay,
        .threshold        = normalized.occupancy_params.threshold,
        .cells_per_update = normalized.occupancy_params.cells_per_update,
        .update_interval  = normalized.occupancy_params.update_interval,
        .warmup_steps     = normalized.occupancy_params.warmup_steps,
        .aabb_min         = float3{normalized.aabb_min.x, normalized.aabb_min.y, normalized.aabb_min.z},
        .aabb_max         = float3{normalized.aabb_max.x, normalized.aabb_max.y, normalized.aabb_max.z},
    };
    const nerf::sampler::SamplerRequest sampler_request{
        .stream                   = runtime->stream,
        .cams                     = reinterpret_cast<const float4*>(ctx->xforms.ptr),
        .images                   = reinterpret_cast<const std::uint8_t*>(ctx->images.ptr),
        .bitfield                 = reinterpret_cast<const std::uint32_t*>(ctx->occupancy_bitfield.ptr),
        .sample_rays              = reinterpret_cast<nerf::sampler::SampleRay*>(ctx->sample_rays.ptr),
        .sample_steps             = reinterpret_cast<nerf::sampler::SampleStep*>(ctx->sample_steps.ptr),
        .batch_state              = reinterpret_cast<nerf::sampler::SampleBatchState*>(ctx->sample_batch_state.ptr),
        .occupancy_grid_res       = ctx->occupancy_grid_res,
        .rays_per_batch           = normalized.rays_per_batch,
        .max_sample_steps_per_ray = normalized.max_sample_steps_per_ray,
        .max_sample_step_count    = static_cast<std::uint32_t>(total_sample_steps),
        .image_width              = ctx->dataset_info.image_width,
        .image_height             = ctx->dataset_info.image_height,
        .aabb_min                 = float3{normalized.aabb_min.x, normalized.aabb_min.y, normalized.aabb_min.z},
        .aabb_max                 = float3{normalized.aabb_max.x, normalized.aabb_max.y, normalized.aabb_max.z},
    };
    const nerf::runtime::TrainingStepRequest training_request{
        .sample_rays           = reinterpret_cast<const nerf::sampler::SampleRay*>(ctx->sample_rays.ptr),
        .sample_steps          = reinterpret_cast<const nerf::sampler::SampleStep*>(ctx->sample_steps.ptr),
        .batch_state           = reinterpret_cast<const nerf::sampler::SampleBatchState*>(ctx->sample_batch_state.ptr),
        .camera_count          = ctx->dataset_info.image_count,
        .max_sample_step_count = static_cast<std::uint32_t>(total_sample_steps),
        .train_cfg =
            nerf::runtime::TrainStepConfig{
                .learning_rate   = normalized.hyper_params.learning_rate,
                .adam_beta1      = normalized.hyper_params.adam_beta1,
                .adam_beta2      = normalized.hyper_params.adam_beta2,
                .adam_eps        = normalized.hyper_params.adam_eps,
                .lr_decay_ksteps = normalized.hyper_params.lr_decay_ksteps,
            },
    };
    if (!nerf::runtime::configure_training_plan(*runtime, occupancy_request, sampler_request, training_request, normalized)) return NERF_STATUS_INTERNAL_ERROR;

    ctx->training_configured = true;
    ctx->training_config     = normalized;
    return NERF_STATUS_OK;
}

NerfStatus nerf_train_step(void* context, const NerfStepRequest* request) {
    if (!context || !request) return NERF_STATUS_INVALID_ARGUMENT;

    const ContextStorage* ctx = static_cast<ContextStorage*>(context);
    if (!ctx->dataset_loaded) return NERF_STATUS_DATASET_NOT_LOADED;
    if (!ctx->training_configured) return NERF_STATUS_TRAINING_NOT_CONFIGURED;
    if (request->rays_per_batch != ctx->training_config.rays_per_batch) return NERF_STATUS_CONFIGURATION_MISMATCH;
    if (request->max_sample_steps_per_ray != ctx->training_config.max_sample_steps_per_ray) return NERF_STATUS_CONFIGURATION_MISMATCH;

    const std::uint64_t total_sample_steps = static_cast<std::uint64_t>(request->rays_per_batch) * static_cast<std::uint64_t>(request->max_sample_steps_per_ray);
    if (total_sample_steps > static_cast<std::uint64_t>(ctx->max_sample_steps)) return NERF_STATUS_RANGE_ERROR;
    if (!nerf::runtime::run_training_plan(ctx->cuda_context)) return NERF_STATUS_INTERNAL_ERROR;
    return NERF_STATUS_OK;
}

NerfStatus nerf_read_train_stats(void* context, NerfTrainStats* out_stats) {
    if (!context || !out_stats) return NERF_STATUS_INVALID_ARGUMENT;
    const ContextStorage* ctx = static_cast<ContextStorage*>(context);
    if (!ctx->training_configured) return NERF_STATUS_TRAINING_NOT_CONFIGURED;
    if (!nerf::runtime::read_training_device_stats(ctx->cuda_context, out_stats)) return NERF_STATUS_INTERNAL_ERROR;
    return NERF_STATUS_OK;
}

NerfStatus nerf_save_network_weights(void* context, const NerfCheckpointFileDesc* desc) {
    if (!context || !desc || !desc->path_utf8) return NERF_STATUS_INVALID_ARGUMENT;

    const ContextStorage* ctx = static_cast<ContextStorage*>(context);
    const auto runtime        = nerf::runtime::runtime_for_context(ctx->cuda_context);
    if (!runtime) return NERF_STATUS_INTERNAL_ERROR;
    nerf::network::NetworkCheckpointLayout layout{};
    if (!nerf::network::describe_network_checkpoint_layout(runtime->network, layout)) return NERF_STATUS_INTERNAL_ERROR;

    std::scoped_lock run_lock(runtime->run_mutex);
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;

    nerf::host::HostCheckpointData checkpoint{};
    checkpoint.density_params_f32.resize(runtime->network.density.params_f32.count);
    checkpoint.color_params_f32.resize(runtime->network.color.params_f32.count);
    if (cudaMemcpy(checkpoint.density_params_f32.data(), runtime->network.density.params_f32.ptr, runtime->network.density.params_f32.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemcpy(checkpoint.color_params_f32.data(), runtime->network.color.params_f32.ptr, runtime->network.color.params_f32.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    return nerf::io::save_network_checkpoint_file(*desc, layout, checkpoint);
}

NerfStatus nerf_load_network_weights(void* context, const NerfCheckpointFileDesc* desc) {
    if (!context || !desc || !desc->path_utf8) return NERF_STATUS_INVALID_ARGUMENT;

    const ContextStorage* ctx = static_cast<ContextStorage*>(context);
    const auto runtime        = nerf::runtime::runtime_for_context(ctx->cuda_context);
    if (!runtime) return NERF_STATUS_INTERNAL_ERROR;
    nerf::network::NetworkCheckpointLayout layout{};
    if (!nerf::network::describe_network_checkpoint_layout(runtime->network, layout)) return NERF_STATUS_INTERNAL_ERROR;

    nerf::host::HostCheckpointData checkpoint{};
    const NerfStatus load_status = nerf::io::load_network_checkpoint_file(*desc, layout, checkpoint);
    if (load_status != NERF_STATUS_OK) return load_status;

    if (checkpoint.density_params_f32.size() != runtime->network.density.params_f32.count) return NERF_STATUS_CHECKPOINT_MISMATCH;
    if (checkpoint.color_params_f32.size() != runtime->network.color.params_f32.count) return NERF_STATUS_CHECKPOINT_MISMATCH;

    std::scoped_lock run_lock(runtime->run_mutex);
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemcpy(runtime->network.density.params_f32.ptr, checkpoint.density_params_f32.data(), runtime->network.density.params_f32.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemcpy(runtime->network.color.params_f32.ptr, checkpoint.color_params_f32.data(), runtime->network.color.params_f32.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;

    constexpr std::uint32_t convert_threads = kConvertThreads;
    nerf::runtime::k_float_to_half<<<(static_cast<std::uint32_t>(runtime->network.density.params.count) + convert_threads - 1u) / convert_threads, convert_threads, 0, runtime->stream>>>(runtime->network.density.params_f32.ptr, runtime->network.density.params.ptr, static_cast<std::uint32_t>(runtime->network.density.params.count));
    nerf::runtime::k_float_to_half<<<(static_cast<std::uint32_t>(runtime->network.color.params.count) + convert_threads - 1u) / convert_threads, convert_threads, 0, runtime->stream>>>(runtime->network.color.params_f32.ptr, runtime->network.color.params.ptr, static_cast<std::uint32_t>(runtime->network.color.params.count));
    if (cudaGetLastError() != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;

    if (cudaMemsetAsync(runtime->network.density.gradients.ptr, 0, runtime->network.density.gradients.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->network.density.gradients_tmp.ptr, 0, runtime->network.density.gradients_tmp.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->network.density.adam_m.ptr, 0, runtime->network.density.adam_m.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->network.density.adam_v.ptr, 0, runtime->network.density.adam_v.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->network.color.gradients.ptr, 0, runtime->network.color.gradients.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->network.color.gradients_tmp.ptr, 0, runtime->network.color.gradients_tmp.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->network.color.adam_m.ptr, 0, runtime->network.color.adam_m.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->network.color.adam_v.ptr, 0, runtime->network.color.adam_v.bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(runtime->device_state, 0, sizeof(nerf::runtime::TrainingDeviceState), runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(ctx->occupancy_bitfield.ptr, 0, ctx->occupancy_bitfield.size_bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaMemsetAsync(ctx->occupancy_density.ptr, 0, ctx->occupancy_density.size_bytes, runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) return NERF_STATUS_CUDA_FAILURE;

    runtime->host_frame_index = 0u;
    return NERF_STATUS_OK;
}
}
