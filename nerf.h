#ifndef NERF_H
#define NERF_H

#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(NERF_BUILD_SHARED) && defined(NERF_BUILD_EXPORTS)
#define NERF_API __declspec(dllexport)
#elif defined(NERF_BUILD_SHARED)
#define NERF_API __declspec(dllimport)
#else
#define NERF_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#if defined(NERF_BUILD_SHARED)
#define NERF_API __attribute__((visibility("default")))
#else
#define NERF_API
#endif
#else
#define NERF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif


typedef enum NerfStatus { NERF_STATUS_OK = 0, NERF_STATUS_INVALID_ARGUMENT = 1, NERF_STATUS_OUT_OF_MEMORY = 2, NERF_STATUS_CUDA_FAILURE = 3, NERF_STATUS_DATASET_NOT_LOADED = 4, NERF_STATUS_TRAINING_NOT_CONFIGURED = 5, NERF_STATUS_RANGE_ERROR = 6, NERF_STATUS_OVERFLOW = 7, NERF_STATUS_INTERNAL_ERROR = 8, NERF_STATUS_CHECKPOINT_INVALID = 16, NERF_STATUS_CHECKPOINT_MISMATCH = 17 } NerfStatus;

typedef struct NerfCreateDesc {
    uint32_t occupancy_grid_res;
    uint32_t max_sample_steps;
    uint32_t max_batch_rays;
    uint64_t arena_alignment_bytes;

    const uint8_t* images_rgba8;
    uint64_t images_bytes;

    // Row-major camera-to-world matrices packed as image_count consecutive 4x4 matrices:
    // [ r00 r01 r02 tx
    //   r10 r11 r12 ty
    //   r20 r21 r22 tz
    //   0   0   0   1 ]
    const float* cameras_4x4_packed;
    uint64_t cameras_bytes;

    uint32_t image_count;
    uint32_t image_width;
    uint32_t image_height;

    float fx;
    float fy;
    float cx;
    float cy;
} NerfCreateDesc;
NERF_API NerfStatus nerf_create_context(const NerfCreateDesc* desc, void** out_context);
NERF_API NerfStatus nerf_destroy_context(void* context);

typedef struct NerfHyperParams {
    float learning_rate;
    float adam_beta1;
    float adam_beta2;
    float adam_eps;
    float grad_clip_norm;
    float update_guard_grad_norm;
    float loss_scale;
    uint32_t lr_decay_ksteps;
} NerfHyperParams;
typedef struct NerfOccupancyParams {
    float decay;
    float threshold;
    uint32_t cells_per_update;
    uint32_t update_interval;
    uint32_t warmup_steps;
} NerfOccupancyParams;
typedef struct NerfTrainingConfig {
    float aabb_min_x;
    float aabb_min_y;
    float aabb_min_z;
    float aabb_max_x;
    float aabb_max_y;
    float aabb_max_z;
    NerfHyperParams hyper_params;
    NerfOccupancyParams occupancy_params;
    uint32_t rays_per_batch;
    uint32_t max_sample_steps_per_ray;
} NerfTrainingConfig;
NERF_API NerfStatus nerf_configure_training(void* context, const NerfTrainingConfig* config);
NERF_API NerfStatus nerf_train_step(void* context);

typedef struct NerfTrainStats {
    float loss;
    float grad_norm;
    uint32_t has_nonfinite;
    uint32_t completed_steps;
    uint32_t last_train_camera;
} NerfTrainStats;
NERF_API NerfStatus nerf_read_train_stats(void* context, NerfTrainStats* out_stats);

typedef enum NerfMemoryKind { NERF_MEMORY_HOST = 1, NERF_MEMORY_CUDA_DEVICE = 2 } NerfMemoryKind;
typedef struct NerfInferenceRequest {
    uint32_t camera_idx;
    uint32_t samples_per_ray;
    uint32_t memory_kind;
    void* dst_rgba8;
    uint64_t dst_bytes;
} NerfInferenceRequest;
typedef struct NerfInferenceInfo {
    uint64_t capacity_bytes;
    uint64_t valid_bytes;
    uint32_t row_stride_bytes;
    uint32_t width;
    uint32_t height;
    uint32_t generation;
} NerfInferenceInfo;
NERF_API NerfStatus nerf_inference(void* context, const NerfInferenceRequest* request, NerfInferenceInfo* out_info);

typedef struct NerfCheckpointFileDesc {
    const char* path_utf8;
} NerfCheckpointFileDesc;
NERF_API NerfStatus nerf_save_network_weights(void* context, const NerfCheckpointFileDesc* desc);
NERF_API NerfStatus nerf_load_network_weights(void* context, const NerfCheckpointFileDesc* desc);

#ifdef __cplusplus
}
#endif

#endif
