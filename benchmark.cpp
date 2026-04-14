#include "nerf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "json/json.hpp"

import std;

constexpr std::uint32_t kDefaultSteps             = 10000u;
constexpr std::uint32_t kDefaultRays              = 4096u;
constexpr std::uint32_t kDefaultMaxSampleSteps    = 64u;
constexpr std::uint32_t kDefaultLogInterval       = 200u;
constexpr std::uint32_t kDefaultInferenceInterval = 2000u;
constexpr std::uint32_t kDefaultInferenceCamera   = 0u;
constexpr std::uint32_t kDefaultInferenceSamples  = 64u;

struct HostDatasetData {
    std::vector<std::uint8_t> images_rgba8;
    std::vector<float> cameras_4x4_packed;
    struct Info {
        uint32_t image_count;
        uint32_t image_width;
        uint32_t image_height;
        float fx;
        float fy;
        float cx;
        float cy;
    } info{};
};

enum class CoordAxis : std::uint32_t {
    PosX = 0u,
    NegX = 1u,
    PosY = 2u,
    NegY = 3u,
    PosZ = 4u,
    NegZ = 5u,
};

struct CoordBasis {
    CoordAxis x;
    CoordAxis y;
    CoordAxis z;
};

static bool parse_u32(std::string_view value, std::uint32_t& out) {
    std::uint64_t parsed = 0u;
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc{} || ptr != value.data() + value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    out = static_cast<std::uint32_t>(parsed);
    return true;
}

static bool safe_mul_u64(const std::uint64_t a, const std::uint64_t b, std::uint64_t* out) {
    if (!out) return false;
    if (a != 0u && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

struct InferenceMetrics {
    double psnr      = 0.0;
    double pred_luma = 0.0;
};

static InferenceMetrics evaluate_inference_rgba8(const HostDatasetData& dataset, const std::uint32_t camera_idx, const std::span<const std::uint8_t> pred_rgba8) {
    const std::uint32_t image_width  = dataset.info.image_width;
    const std::uint32_t image_height = dataset.info.image_height;
    const std::size_t image_stride   = static_cast<std::size_t>(image_width) * static_cast<std::size_t>(image_height) * 4u;
    const std::size_t camera_base    = static_cast<std::size_t>(camera_idx) * image_stride;
    constexpr float inv255           = 1.0f / 255.0f;
    double mse                       = 0.0;
    double luma_sum                  = 0.0;

    for (std::uint32_t y = 0u; y < image_height; ++y) {
        for (std::uint32_t x = 0u; x < image_width; ++x) {
            const std::size_t gt_base = camera_base + (static_cast<std::size_t>(y) * image_width + x) * 4u;
            const float ga            = static_cast<float>(dataset.images_rgba8[gt_base + 3u]) * inv255;
            const float gr0           = static_cast<float>(dataset.images_rgba8[gt_base + 0u]) * inv255;
            const float gg0           = static_cast<float>(dataset.images_rgba8[gt_base + 1u]) * inv255;
            const float gb0           = static_cast<float>(dataset.images_rgba8[gt_base + 2u]) * inv255;

            const std::size_t pred_base = (static_cast<std::size_t>(y) * image_width + x) * 4u;
            const float pr              = static_cast<float>(pred_rgba8[pred_base + 0u]) * inv255;
            const float pg              = static_cast<float>(pred_rgba8[pred_base + 1u]) * inv255;
            const float pb              = static_cast<float>(pred_rgba8[pred_base + 2u]) * inv255;

            const float gt_r = gr0 * ga + (1.0f - ga);
            const float gt_g = gg0 * ga + (1.0f - ga);
            const float gt_b = gb0 * ga + (1.0f - ga);

            const double dr = pr - gt_r;
            const double dg = pg - gt_g;
            const double db = pb - gt_b;
            mse += (dr * dr + dg * dg + db * db) / 3.0;
            luma_sum += static_cast<double>(0.2126f * pr + 0.7152f * pg + 0.0722f * pb);
        }
    }

    mse /= static_cast<double>(image_width) * static_cast<double>(image_height);
    return InferenceMetrics{
        .psnr      = mse > 0.0 ? -10.0 * std::log10(mse) : std::numeric_limits<double>::infinity(),
        .pred_luma = luma_sum / (static_cast<double>(image_width) * static_cast<double>(image_height)),
    };
}

static bool write_rgba_png(const std::filesystem::path& path, const std::uint32_t width, const std::uint32_t height, const std::span<const std::uint8_t> rgba8) {
    std::error_code ec{};
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    return stbi_write_png(path.string().c_str(), static_cast<int>(width), static_cast<int>(height), 4, rgba8.data(), static_cast<int>(width) * 4) != 0;
}

static NerfStatus resolve_dataset_json_path(const char* path_utf8, std::filesystem::path& out_json_path) {
    if (!path_utf8) return NERF_STATUS_INVALID_ARGUMENT;
    std::filesystem::path root{path_utf8};
    std::filesystem::path json_path = root;
    if (!json_path.has_extension() || json_path.extension() != ".json") json_path = root / "transforms_train.json";
    std::error_code ec{};
    json_path = std::filesystem::absolute(json_path, ec);
    if (ec) return NERF_STATUS_INTERNAL_ERROR;
    json_path = std::filesystem::weakly_canonical(json_path, ec);
    if (ec) return NERF_STATUS_INTERNAL_ERROR;
    out_json_path = std::move(json_path);
    return NERF_STATUS_OK;
}

static bool axis_to_basis_entry(const CoordAxis axis, std::uint32_t* out_axis, float* out_sign) {
    if (!out_axis || !out_sign) return false;
    switch (axis) {
    case CoordAxis::PosX:
        *out_axis = 0u;
        *out_sign = 1.0f;
        return true;
    case CoordAxis::NegX:
        *out_axis = 0u;
        *out_sign = -1.0f;
        return true;
    case CoordAxis::PosY:
        *out_axis = 1u;
        *out_sign = 1.0f;
        return true;
    case CoordAxis::NegY:
        *out_axis = 1u;
        *out_sign = -1.0f;
        return true;
    case CoordAxis::PosZ:
        *out_axis = 2u;
        *out_sign = 1.0f;
        return true;
    case CoordAxis::NegZ:
        *out_axis = 2u;
        *out_sign = -1.0f;
        return true;
    }
    return false;
}

static NerfStatus fill_basis_matrix(const CoordBasis& basis, float out_matrix[9]) {
    std::fill_n(out_matrix, 9, 0.0f);
    const CoordAxis dirs[3] = {basis.x, basis.y, basis.z};
    std::uint32_t seen_axes = 0u;
    for (std::size_t c = 0u; c < 3u; ++c) {
        std::uint32_t axis = 0u;
        float sign         = 0.0f;
        if (!axis_to_basis_entry(dirs[c], &axis, &sign)) return NERF_STATUS_INVALID_ARGUMENT;
        if ((seen_axes & (1u << axis)) != 0u) return NERF_STATUS_INVALID_ARGUMENT;
        seen_axes |= (1u << axis);
        out_matrix[axis * 3u + c] = sign;
    }
    return seen_axes == 0x7u ? NERF_STATUS_OK : NERF_STATUS_INVALID_ARGUMENT;
}

static NerfStatus load_nerf_synthetic_host_dataset(const char* path_utf8, HostDatasetData& out_data) {
    constexpr CoordBasis src_world{.x = CoordAxis::PosX, .y = CoordAxis::NegZ, .z = CoordAxis::PosY};
    constexpr CoordBasis src_camera{.x = CoordAxis::PosX, .y = CoordAxis::PosY, .z = CoordAxis::PosZ};
    constexpr CoordBasis dst_world{.x = CoordAxis::PosX, .y = CoordAxis::PosY, .z = CoordAxis::PosZ};
    constexpr CoordBasis dst_camera{.x = CoordAxis::PosX, .y = CoordAxis::PosY, .z = CoordAxis::PosZ};
    constexpr float kScale = 0.33f;
    constexpr float kOffX  = 0.5f;
    constexpr float kOffY  = 0.5f;
    constexpr float kOffZ  = 0.5f;

    std::filesystem::path json_path{};
    NerfStatus status = resolve_dataset_json_path(path_utf8, json_path);
    if (status != NERF_STATUS_OK) return status;

    std::ifstream json_file(json_path, std::ios::binary);
    if (!json_file) return NERF_STATUS_INTERNAL_ERROR;

    nlohmann::json json_value;
    try {
        json_file >> json_value;
    } catch (...) {
        return NERF_STATUS_INTERNAL_ERROR;
    }

    if (!json_value.contains("camera_angle_x") || !json_value.contains("frames")) return NERF_STATUS_INTERNAL_ERROR;
    if (!json_value.at("frames").is_array() || json_value.at("frames").empty()) return NERF_STATUS_INTERNAL_ERROR;

    float camera_angle_x = 0.0f;
    try {
        camera_angle_x = json_value.at("camera_angle_x").get<float>();
    } catch (...) {
        return NERF_STATUS_INTERNAL_ERROR;
    }
    if (!std::isfinite(camera_angle_x)) return NERF_STATUS_INTERNAL_ERROR;

    const nlohmann::json& frames  = json_value.at("frames");
    const std::size_t frame_count = frames.size();
    if (frame_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return NERF_STATUS_OVERFLOW;

    std::vector<std::filesystem::path> image_paths(frame_count);
    std::vector<float> row_major_4x4(frame_count * 16u, 0.0f);
    for (std::size_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
        const nlohmann::json& frame = frames[frame_index];
        if (!frame.contains("file_path") || !frame.contains("transform_matrix")) return NERF_STATUS_INTERNAL_ERROR;

        try {
            std::filesystem::path image_path = json_path.parent_path() / frame.at("file_path").get<std::string>();
            if (!image_path.has_extension()) image_path.replace_extension(".png");
            std::error_code ec{};
            image_path = std::filesystem::absolute(image_path, ec);
            if (ec) return NERF_STATUS_INTERNAL_ERROR;
            image_path = std::filesystem::weakly_canonical(image_path, ec);
            if (ec) return NERF_STATUS_INTERNAL_ERROR;
            image_paths[frame_index] = std::move(image_path);
        } catch (...) {
            return NERF_STATUS_INTERNAL_ERROR;
        }

        try {
            for (std::size_t r = 0u; r < 4u; ++r) {
                for (std::size_t c = 0u; c < 4u; ++c) {
                    const float value = frame.at("transform_matrix").at(r).at(c).get<float>();
                    if (!std::isfinite(value)) return NERF_STATUS_INTERNAL_ERROR;
                    row_major_4x4[frame_index * 16u + r * 4u + c] = value;
                }
            }
        } catch (...) {
            return NERF_STATUS_INTERNAL_ERROR;
        }
    }

    int first_width       = 0;
    int first_height      = 0;
    stbi_uc* first_pixels = stbi_load(image_paths[0].string().c_str(), &first_width, &first_height, nullptr, 4);
    if (!first_pixels || first_width <= 0 || first_height <= 0) {
        if (first_pixels) stbi_image_free(first_pixels);
        return NERF_STATUS_INTERNAL_ERROR;
    }

    std::uint64_t bytes_per_image = static_cast<std::uint64_t>(first_width);
    if (!safe_mul_u64(bytes_per_image, static_cast<std::uint64_t>(first_height), &bytes_per_image)) {
        stbi_image_free(first_pixels);
        return NERF_STATUS_OVERFLOW;
    }
    if (!safe_mul_u64(bytes_per_image, 4u, &bytes_per_image)) {
        stbi_image_free(first_pixels);
        return NERF_STATUS_OVERFLOW;
    }

    std::uint64_t total_image_bytes = bytes_per_image;
    if (!safe_mul_u64(total_image_bytes, frame_count, &total_image_bytes)) {
        stbi_image_free(first_pixels);
        return NERF_STATUS_OVERFLOW;
    }

    std::vector<std::uint8_t> host_images_rgba8(total_image_bytes);
    std::memcpy(host_images_rgba8.data(), first_pixels, bytes_per_image);
    stbi_image_free(first_pixels);

    for (std::size_t image_index = 1u; image_index < frame_count; ++image_index) {
        int width       = 0;
        int height      = 0;
        stbi_uc* pixels = stbi_load(image_paths[image_index].string().c_str(), &width, &height, nullptr, 4);
        if (!pixels) return NERF_STATUS_INTERNAL_ERROR;
        if (width != first_width || height != first_height) {
            stbi_image_free(pixels);
            return NERF_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(host_images_rgba8.data() + image_index * bytes_per_image, pixels, bytes_per_image);
        stbi_image_free(pixels);
    }

    const float width_f  = static_cast<float>(first_width);
    const float height_f = static_cast<float>(first_height);
    const float fx       = 0.5f * width_f / std::tan(camera_angle_x * 0.5f);
    const float fy       = fx;
    const float cx       = 0.5f * (width_f - 1.0f);
    const float cy       = 0.5f * (height_f - 1.0f);

    float src_world_basis[9]{};
    float src_camera_basis[9]{};
    float dst_world_basis[9]{};
    float dst_camera_basis[9]{};
    status = fill_basis_matrix(src_world, src_world_basis);
    if (status != NERF_STATUS_OK) return status;
    status = fill_basis_matrix(src_camera, src_camera_basis);
    if (status != NERF_STATUS_OK) return status;
    status = fill_basis_matrix(dst_world, dst_world_basis);
    if (status != NERF_STATUS_OK) return status;
    status = fill_basis_matrix(dst_camera, dst_camera_basis);
    if (status != NERF_STATUS_OK) return status;

    float world_map[9]{};
    float camera_map[9]{};
    for (std::size_t r = 0u; r < 3u; ++r) {
        for (std::size_t c = 0u; c < 3u; ++c) {
            for (std::size_t k = 0u; k < 3u; ++k) {
                world_map[r * 3u + c] += dst_world_basis[k * 3u + r] * src_world_basis[k * 3u + c];
                camera_map[r * 3u + c] += src_camera_basis[k * 3u + r] * dst_camera_basis[k * 3u + c];
            }
        }
    }

    std::vector<float> packed_cameras(frame_count * 16u, 0.0f);
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

        float tmp_rot[9]{};
        float dst_rot[9]{};
        float dst_translation[3]{};
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

        float* packed = packed_cameras.data() + frame_index * 16u;
        packed[0]     = dst_rot[0];
        packed[1]     = dst_rot[1];
        packed[2]     = dst_rot[2];
        packed[3]     = dst_translation[0] * kScale + kOffX;
        packed[4]     = dst_rot[3];
        packed[5]     = dst_rot[4];
        packed[6]     = dst_rot[5];
        packed[7]     = dst_translation[1] * kScale + kOffY;
        packed[8]     = dst_rot[6];
        packed[9]     = dst_rot[7];
        packed[10]    = dst_rot[8];
        packed[11]    = dst_translation[2] * kScale + kOffZ;
        packed[12]    = 0.0f;
        packed[13]    = 0.0f;
        packed[14]    = 0.0f;
        packed[15]    = 1.0f;
    }

    out_data.images_rgba8       = std::move(host_images_rgba8);
    out_data.cameras_4x4_packed = std::move(packed_cameras);
    out_data.info               = HostDatasetData::Info{
                      .image_count  = static_cast<std::uint32_t>(frame_count),
                      .image_width  = static_cast<std::uint32_t>(first_width),
                      .image_height = static_cast<std::uint32_t>(first_height),
                      .fx           = fx,
                      .fy           = fy,
                      .cx           = cx,
                      .cy           = cy,
    };
    return NERF_STATUS_OK;
}

int main(int argc, char** argv) {
    std::filesystem::path dataset_path{};
    std::filesystem::path load_weights_path{};
    std::filesystem::path save_weights_path{};
    std::filesystem::path inference_out_path{"nerf_inference_final.png"};
    std::uint32_t steps              = kDefaultSteps;
    std::uint32_t rays_per_batch     = kDefaultRays;
    std::uint32_t max_sample_steps   = kDefaultMaxSampleSteps;
    std::uint32_t log_interval       = kDefaultLogInterval;
    std::uint32_t inference_interval = kDefaultInferenceInterval;
    std::uint32_t inference_camera   = kDefaultInferenceCamera;
    std::uint32_t inference_samples  = kDefaultInferenceSamples;

    auto print_help = [&]() {
        const std::string exe_name = (argc > 0 && argv[0]) ? std::filesystem::path(argv[0]).filename().string() : std::string{"benchmark"};
        std::cout << "Usage: " << exe_name << " --dataset <path> [options]\n";
        std::cout << "  --steps <N> default " << steps << '\n';
        std::cout << "  --rays <N> default " << rays_per_batch << '\n';
        std::cout << "  --max-sample-steps <N> default " << max_sample_steps << '\n';
        std::cout << "  --log-interval <N> default " << log_interval << '\n';
        std::cout << "  --inference-interval <N> default " << inference_interval << '\n';
        std::cout << "  --inference-camera <N> default " << inference_camera << '\n';
        std::cout << "  --inference-samples <N> default " << inference_samples << '\n';
        std::cout << "  --inference-out <path> default " << inference_out_path.string() << '\n';
        std::cout << "  --load-weights <path> optional\n";
        std::cout << "  --save-weights <path> optional\n";
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key == "--help" || key == "-h") {
            print_help();
            return 0;
        }
        if (i + 1 >= argc) {
            std::cerr << "missing value for " << key << '\n';
            return 2;
        }

        const std::string_view value = argv[++i];
        if (key == "--dataset") {
            dataset_path = value;
        } else if (key == "--load-weights") {
            load_weights_path = value;
        } else if (key == "--save-weights") {
            save_weights_path = value;
        } else if (key == "--steps") {
            if (!parse_u32(value, steps)) {
                std::cerr << "invalid value for --steps: " << value << '\n';
                return 2;
            }
        } else if (key == "--rays") {
            if (!parse_u32(value, rays_per_batch)) {
                std::cerr << "invalid value for --rays: " << value << '\n';
                return 2;
            }
        } else if (key == "--max-sample-steps") {
            if (!parse_u32(value, max_sample_steps)) {
                std::cerr << "invalid value for --max-sample-steps: " << value << '\n';
                return 2;
            }
        } else if (key == "--log-interval") {
            if (!parse_u32(value, log_interval)) {
                std::cerr << "invalid value for --log-interval: " << value << '\n';
                return 2;
            }
        } else if (key == "--inference-interval") {
            if (!parse_u32(value, inference_interval)) {
                std::cerr << "invalid value for --inference-interval: " << value << '\n';
                return 2;
            }
        } else if (key == "--inference-camera") {
            if (!parse_u32(value, inference_camera)) {
                std::cerr << "invalid value for --inference-camera: " << value << '\n';
                return 2;
            }
        } else if (key == "--inference-samples") {
            if (!parse_u32(value, inference_samples)) {
                std::cerr << "invalid value for --inference-samples: " << value << '\n';
                return 2;
            }
        } else if (key == "--inference-out") {
            inference_out_path = value;
        } else {
            std::cerr << "unknown argument: " << key << '\n';
            return 2;
        }
    }

    if (dataset_path.empty()) {
        std::cerr << "--dataset is required\n";
        return 2;
    }
    if (steps == 0u || rays_per_batch == 0u || max_sample_steps == 0u || log_interval == 0u || inference_interval == 0u || inference_samples == 0u) {
        std::cerr << "steps/rays/max-sample-steps/log-interval/inference-interval/inference-samples must be > 0\n";
        return 2;
    }

    void* context = nullptr;
    struct ContextGuard {
        void** context = nullptr;
        ~ContextGuard() {
            if (!context || !*context) return;
            (void) nerf_destroy_context(*context);
            *context = nullptr;
        }
    } destroy_context_scope{.context = &context};

    constexpr NerfHyperParams train_hp{
        .learning_rate   = 5e-4f,
        .adam_beta1      = 0.9f,
        .adam_beta2      = 0.999f,
        .adam_eps        = 1e-8f,
        .lr_decay_ksteps = 250u,
    };
    constexpr NerfOccupancyParams occupancy_hp{
        .decay            = 0.98f,
        .threshold        = 0.01f,
        .cells_per_update = 65536u,
        .update_interval  = 1u,
        .warmup_steps     = 32u,
    };

    try {
        HostDatasetData host_data{};
        const std::string dataset_path_utf8 = dataset_path.string();
        NerfStatus status                   = load_nerf_synthetic_host_dataset(dataset_path_utf8.c_str(), host_data);
        if (status != NERF_STATUS_OK) throw std::runtime_error("load_nerf_synthetic_host_dataset failed: status=" + std::to_string(status));
        if (inference_camera >= host_data.info.image_count) throw std::runtime_error("inference camera is out of range");

        std::uint64_t inference_bytes = host_data.info.image_width;
        if (!safe_mul_u64(inference_bytes, host_data.info.image_height, &inference_bytes)) throw std::runtime_error("inference image size overflow");
        if (!safe_mul_u64(inference_bytes, 4u, &inference_bytes)) throw std::runtime_error("inference image size overflow");
        std::vector<std::uint8_t> inference_rgba8(inference_bytes);

        NerfCreateDesc create_desc{
            .occupancy_grid_res    = 128u,
            .max_sample_steps      = 1u << 21,
            .max_batch_rays        = 1u << 16,
            .arena_alignment_bytes = 256u,
            .images_rgba8          = host_data.images_rgba8.data(),
            .images_bytes          = (host_data.images_rgba8.size()),
            .cameras_4x4_packed    = host_data.cameras_4x4_packed.data(),
            .cameras_bytes         = host_data.cameras_4x4_packed.size() * sizeof(float),
            .image_count           = host_data.info.image_count,
            .image_width           = host_data.info.image_width,
            .image_height          = host_data.info.image_height,
            .fx                    = host_data.info.fx,
            .fy                    = host_data.info.fy,
            .cx                    = host_data.info.cx,
            .cy                    = host_data.info.cy,
        };
        status = nerf_create_context(&create_desc, &context);
        if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_create_context failed: status=" + std::to_string(status));

        if (!load_weights_path.empty()) {
            const std::string load_path_utf8 = load_weights_path.string();
            const NerfCheckpointFileDesc checkpoint_desc{.path_utf8 = load_path_utf8.c_str()};
            status = nerf_load_network_weights(context, &checkpoint_desc);
            if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_load_network_weights failed: status=" + std::to_string(status));
        }

        const NerfTrainingConfig training_config{
            .aabb_min_x               = 0.0f,
            .aabb_min_y               = 0.0f,
            .aabb_min_z               = 0.0f,
            .aabb_max_x               = 1.0f,
            .aabb_max_y               = 1.0f,
            .aabb_max_z               = 1.0f,
            .hyper_params             = train_hp,
            .occupancy_params         = occupancy_hp,
            .rays_per_batch           = rays_per_batch,
            .max_sample_steps_per_ray = max_sample_steps,
        };
        status = nerf_configure_training(context, &training_config);
        if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_configure_training failed: status=" + std::to_string(status));

        const auto t0  = std::chrono::steady_clock::now();
        float loss_ema = 0.0f;
        for (std::uint32_t step = 0u; step < steps; ++step) {
            status = nerf_train_step(context);
            if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_train_step failed: status=" + std::to_string(status));

            if ((step + 1u) % log_interval == 0u || (step + 1u) == steps) {
                NerfTrainStats train_stats{};
                status = nerf_read_train_stats(context, &train_stats);
                if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_read_train_stats failed: status=" + std::to_string(status));

                const float loss          = train_stats.loss;
                const float grad_norm     = train_stats.grad_norm;
                const bool has_nonfinite  = train_stats.has_nonfinite != 0u;
                loss_ema                  = step == 0u ? loss : (0.95f * loss_ema + 0.05f * loss);
                const float sec           = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
                const float steps_per_sec = sec > 0.0f ? static_cast<float>(step + 1u) / sec : 0.0f;
                const double eta_sec      = steps_per_sec > 0.0f ? static_cast<double>(steps - (step + 1u)) / static_cast<double>(steps_per_sec) : std::numeric_limits<double>::infinity();
                std::cout << "[train] step=" << (step + 1u) << " loss=" << loss << " loss_ema=" << loss_ema << " grad_norm=" << grad_norm << " sps=" << steps_per_sec << " eta_sec=" << eta_sec << " nonfinite=" << (has_nonfinite ? 1 : 0) << '\n';
            }

            if ((step + 1u) % inference_interval == 0u || (step + 1u) == steps) {
                NerfInferenceInfo inference_info{};
                const NerfInferenceRequest inference_request{
                    .camera_idx      = inference_camera,
                    .samples_per_ray = inference_samples,
                    .memory_kind     = NERF_MEMORY_HOST,
                    .dst_rgba8       = inference_rgba8.data(),
                    .dst_bytes       = (inference_rgba8.size()),
                };
                status = nerf_inference(context, &inference_request, &inference_info);
                if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_inference failed: status=" + std::to_string(status));

                const InferenceMetrics metrics = evaluate_inference_rgba8(host_data, inference_camera, inference_rgba8);
                std::cout << "[eval] step=" << (step + 1u) << " camera=" << inference_camera << " psnr=" << metrics.psnr << " pred_luma=" << metrics.pred_luma << '\n';

                if ((step + 1u) == steps) {
                    if (!write_rgba_png(inference_out_path, inference_info.width, inference_info.height, inference_rgba8)) throw std::runtime_error("stbi_write_png failed");
                    std::cout << "[eval] saved=" << inference_out_path.string() << '\n';
                }
            }
        }

        if (!save_weights_path.empty()) {
            const std::string save_path_utf8 = save_weights_path.string();
            const NerfCheckpointFileDesc checkpoint_desc{.path_utf8 = save_path_utf8.c_str()};
            status = nerf_save_network_weights(context, &checkpoint_desc);
            if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_save_network_weights failed: status=" + std::to_string(status));
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
