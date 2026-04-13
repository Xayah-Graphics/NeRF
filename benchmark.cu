#include "nerf.h"
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

constexpr std::uint32_t kDefaultSteps          = 10000u;
constexpr std::uint32_t kDefaultRays           = 4096u;
constexpr std::uint32_t kDefaultMaxSampleSteps = 64u;
constexpr std::uint32_t kDefaultLogInterval    = 200u;

static bool parse_u32(std::string_view value, std::uint32_t& out) {
    std::uint64_t parsed = 0u;
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc{} || ptr != value.data() + value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    out = static_cast<std::uint32_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    std::filesystem::path dataset_path{};
    std::filesystem::path load_weights_path{};
    std::filesystem::path save_weights_path{};
    std::uint32_t steps            = kDefaultSteps;
    std::uint32_t rays_per_batch   = kDefaultRays;
    std::uint32_t max_sample_steps = kDefaultMaxSampleSteps;
    std::uint32_t log_interval     = kDefaultLogInterval;

    auto print_help = [&]() {
        const std::string exe_name = (argc > 0 && argv[0]) ? std::filesystem::path(argv[0]).filename().string() : std::string{"benchmark"};
        std::cout << "Usage: " << exe_name << " --dataset <path> [options]\n";
        std::cout << "  --steps <N> default " << steps << '\n';
        std::cout << "  --rays <N> default " << rays_per_batch << '\n';
        std::cout << "  --max-sample-steps <N> default " << max_sample_steps << '\n';
        std::cout << "  --log-interval <N> default " << log_interval << '\n';
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
        } else {
            std::cerr << "unknown argument: " << key << '\n';
            return 2;
        }
    }

    if (dataset_path.empty()) {
        std::cerr << "--dataset is required\n";
        return 2;
    }
    if (steps == 0u || rays_per_batch == 0u || max_sample_steps == 0u || log_interval == 0u) {
        std::cerr << "steps/rays/max-sample-steps/log-interval must be > 0\n";
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

    constexpr NerfCreateDesc create_desc{
        .occupancy_grid_res    = 128u,
        .max_sample_steps      = 1u << 21,
        .max_batch_rays        = 1u << 16,
        .arena_alignment_bytes = 256u,
    };
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
    constexpr NerfCoordSystem blender_scene{
        .world  = {.x = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_X), .y = static_cast<std::uint32_t>(NERF_COORD_AXIS_NEGATIVE_Z), .z = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_Y)},
        .camera = {.x = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_X), .y = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_Y), .z = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_Z)},
    };
    constexpr NerfCoordSystem engine{
        .world  = {.x = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_X), .y = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_Y), .z = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_Z)},
        .camera = {.x = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_X), .y = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_Y), .z = static_cast<std::uint32_t>(NERF_COORD_AXIS_POSITIVE_Z)},
    };

    try {
        NerfStatus status = nerf_create_context(&create_desc, &context);
        if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_create_context failed: status=" + std::to_string(status));

        const std::string dataset_path_utf8 = dataset_path.string();
        const NerfDatasetLoadDesc load_desc{
            .path_utf8     = dataset_path_utf8.c_str(),
            .offset        = NerfVec3{0.5f, 0.5f, 0.5f},
            .scale         = 0.33f,
            .source_system = blender_scene,
            .target_system = engine,
        };
        status = nerf_load_dataset(context, &load_desc, nullptr);
        if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_load_dataset failed: status=" + std::to_string(status));

        if (!load_weights_path.empty()) {
            const std::string load_path_utf8 = load_weights_path.string();
            const NerfCheckpointFileDesc checkpoint_desc{.path_utf8 = load_path_utf8.c_str()};
            status = nerf_load_network_weights(context, &checkpoint_desc);
            if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_load_network_weights failed: status=" + std::to_string(status));
        }

        const NerfTrainingConfig training_config{
            .aabb_min                 = NerfVec3{0.0f, 0.0f, 0.0f},
            .aabb_max                 = NerfVec3{1.0f, 1.0f, 1.0f},
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
