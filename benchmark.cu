#include "nerf.h"
#include <array>
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

enum class CliKey : std::uint8_t {
    dataset,
    steps,
    rays,
    max_sample_steps,
    log_interval,
    load_weights,
    save_weights,
    help,
    unknown,
};

int main(int argc, char** argv) {
    struct BenchmarkConfig {
        std::filesystem::path dataset_path{};
        std::uint32_t steps                    = kDefaultSteps;
        std::uint32_t rays_per_batch           = kDefaultRays;
        std::uint32_t max_sample_steps_per_ray = kDefaultMaxSampleSteps;
        std::uint32_t log_interval             = kDefaultLogInterval;
        NerfVec3 aabb_min{0.0f, 0.0f, 0.0f};
        NerfVec3 aabb_max{1.0f, 1.0f, 1.0f};
        std::filesystem::path load_weights_path{};
        std::filesystem::path save_weights_path{};
    } config{};

    struct CliOptionDesc {
        std::string_view name{};
        CliKey key       = CliKey::unknown;
        bool needs_value = false;
    };

    constexpr std::array<CliOptionDesc, 7> cli_desc{{
        {.name = "--dataset", .key = CliKey::dataset, .needs_value = true},
        {.name = "--steps", .key = CliKey::steps, .needs_value = true},
        {.name = "--rays", .key = CliKey::rays, .needs_value = true},
        {.name = "--max-sample-steps", .key = CliKey::max_sample_steps, .needs_value = true},
        {.name = "--log-interval", .key = CliKey::log_interval, .needs_value = true},
        {.name = "--load-weights", .key = CliKey::load_weights, .needs_value = true},
        {.name = "--save-weights", .key = CliKey::save_weights, .needs_value = true},
    }};

    for (int i = 1; i < argc; ++i) {
        const std::string_view raw_key = argv[i];
        CliOptionDesc option{};
        option.key = CliKey::unknown;
        for (const CliOptionDesc& desc : cli_desc) {
            if (desc.name == raw_key) {
                option = desc;
                break;
            }
        }
        if (raw_key == "--help" || raw_key == "-h") option = CliOptionDesc{.name = "--help", .key = CliKey::help, .needs_value = false};

        if (option.key == CliKey::help) {
            std::string exe_name = (argc > 0 && argv[0]) ? std::filesystem::path(argv[0]).filename().string() : std::string{"benchmark"};
            std::cout << "Usage: " << exe_name << " --dataset <path> [options]\n";
            std::cout << "  --steps <N> default " << config.steps << '\n';
            std::cout << "  --rays <N> default " << config.rays_per_batch << '\n';
            std::cout << "  --max-sample-steps <N> default " << config.max_sample_steps_per_ray << '\n';
            std::cout << "  --log-interval <N> default " << config.log_interval << '\n';
            std::cout << "  --load-weights <path> optional\n";
            std::cout << "  --save-weights <path> optional\n";
            return 0;
        }

        if (option.key == CliKey::unknown) {
            std::cerr << "unknown argument: " << raw_key << '\n';
            return 2;
        }
        if (option.needs_value && i + 1 >= argc) {
            std::cerr << "missing value for " << option.name << '\n';
            return 2;
        }

        const std::string_view value = option.needs_value ? std::string_view{argv[++i]} : std::string_view{};
        switch (option.key) {
        case CliKey::dataset: config.dataset_path = value; break;
        case CliKey::steps:
            {
                std::uint64_t parsed = 0u;
                const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (ec != std::errc{} || ptr != value.data() + value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
                    std::cerr << "invalid value for --steps: " << value << '\n';
                    return 2;
                }
                config.steps = static_cast<std::uint32_t>(parsed);
                break;
            }
        case CliKey::rays:
            {
                std::uint64_t parsed = 0u;
                const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (ec != std::errc{} || ptr != value.data() + value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
                    std::cerr << "invalid value for --rays: " << value << '\n';
                    return 2;
                }
                config.rays_per_batch = static_cast<std::uint32_t>(parsed);
                break;
            }
        case CliKey::max_sample_steps:
            {
                std::uint64_t parsed = 0u;
                const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (ec != std::errc{} || ptr != value.data() + value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
                    std::cerr << "invalid value for --max-sample-steps: " << value << '\n';
                    return 2;
                }
                config.max_sample_steps_per_ray = static_cast<std::uint32_t>(parsed);
                break;
            }
        case CliKey::log_interval:
            {
                std::uint64_t parsed = 0u;
                const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (ec != std::errc{} || ptr != value.data() + value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
                    std::cerr << "invalid value for --log-interval: " << value << '\n';
                    return 2;
                }
                config.log_interval = static_cast<std::uint32_t>(parsed);
                break;
            }
        case CliKey::load_weights: config.load_weights_path = value; break;
        case CliKey::save_weights: config.save_weights_path = value; break;
        case CliKey::help:
        case CliKey::unknown: break;
        }
    }

    if (config.dataset_path.empty()) {
        std::cerr << "--dataset is required\n";
        return 2;
    }
    if (config.steps == 0u || config.rays_per_batch == 0u || config.max_sample_steps_per_ray == 0u || config.log_interval == 0u) {
        std::cerr << "steps/rays/max-sample-steps/log-interval must be > 0\n";
        return 2;
    }

    struct BenchmarkState {
        NerfDatasetInfo dataset_info{};
        void* context = nullptr;
    } state{};

    struct ContextGuard {
        void** context = nullptr;
        ~ContextGuard() {
            if (!context || !*context) return;
            (void) nerf_destroy_context(*context);
            *context = nullptr;
        }
    } destroy_context_scope{.context = &state.context};

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
        NerfStatus status = nerf_create_context(&create_desc, &state.context);
        if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_create_context failed: status=" + std::to_string(status));

        const std::string dataset_path_utf8 = config.dataset_path.string();
        const NerfDatasetLoadDesc load_desc{
            .path_utf8     = dataset_path_utf8.c_str(),
            .offset        = NerfVec3{0.5f, 0.5f, 0.5f},
            .scale         = 0.33f,
            .source_system = blender_scene,
            .target_system = engine,
        };
        status = nerf_load_dataset(state.context, &load_desc, &state.dataset_info);
        if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_load_dataset failed: status=" + std::to_string(status));

        if (!config.load_weights_path.empty()) {
            const std::string load_path_utf8 = config.load_weights_path.string();
            const NerfCheckpointFileDesc checkpoint_desc{.path_utf8 = load_path_utf8.c_str()};
            status = nerf_load_network_weights(state.context, &checkpoint_desc);
            if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_load_network_weights failed: status=" + std::to_string(status));
        }

        const NerfTrainingConfig training_config{
            .aabb_min                 = config.aabb_min,
            .aabb_max                 = config.aabb_max,
            .hyper_params             = train_hp,
            .occupancy_params         = occupancy_hp,
            .rays_per_batch           = config.rays_per_batch,
            .max_sample_steps_per_ray = config.max_sample_steps_per_ray,
        };
        status = nerf_configure_training(state.context, &training_config);
        if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_configure_training failed: status=" + std::to_string(status));

        const auto t0  = std::chrono::steady_clock::now();
        float loss_ema = 0.0f;

        struct StepState {
            float loss          = 0.0f;
            float grad_norm     = 0.0f;
            bool has_nonfinite  = false;
            float sec           = 0.0f;
            float steps_per_sec = 0.0f;
            double eta_sec      = std::numeric_limits<double>::infinity();
        };

        for (std::uint32_t step = 0u; step < config.steps; ++step) {
            const NerfStepRequest train_request{
                .rays_per_batch           = config.rays_per_batch,
                .max_sample_steps_per_ray = config.max_sample_steps_per_ray,
            };
            status = nerf_train_step(state.context, &train_request);
            if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_train_step failed: status=" + std::to_string(status));

            StepState step_state{};
            NerfTrainStats train_stats{};
            if ((step + 1u) % config.log_interval == 0u || (step + 1u) == config.steps) {
                status = nerf_read_train_stats(state.context, &train_stats);
                if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_read_train_stats failed: status=" + std::to_string(status));
                step_state.loss          = train_stats.loss;
                step_state.grad_norm     = train_stats.grad_norm;
                step_state.has_nonfinite = train_stats.has_nonfinite != 0u;
                loss_ema                 = step == 0u ? step_state.loss : (0.95f * loss_ema + 0.05f * step_state.loss);
                const auto now           = std::chrono::steady_clock::now();
                step_state.sec           = std::chrono::duration<float>(now - t0).count();
                step_state.steps_per_sec = step_state.sec > 0.0f ? static_cast<float>(step + 1u) / step_state.sec : 0.0f;
                if (step_state.steps_per_sec > 0.0f) step_state.eta_sec = static_cast<double>(config.steps - (step + 1u)) / static_cast<double>(step_state.steps_per_sec);
                std::cout << "[train] step=" << (step + 1u) << " loss=" << step_state.loss << " loss_ema=" << loss_ema << " grad_norm=" << step_state.grad_norm << " sps=" << step_state.steps_per_sec << " eta_sec=" << step_state.eta_sec << " nonfinite=" << (step_state.has_nonfinite ? 1 : 0) << '\n';
            }
        }

        if (!config.save_weights_path.empty()) {
            const std::string save_path_utf8 = config.save_weights_path.string();
            const NerfCheckpointFileDesc checkpoint_desc{.path_utf8 = save_path_utf8.c_str()};
            status = nerf_save_network_weights(state.context, &checkpoint_desc);
            if (status != NERF_STATUS_OK) throw std::runtime_error("nerf_save_network_weights failed: status=" + std::to_string(status));
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
