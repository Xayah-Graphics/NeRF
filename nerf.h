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

#define NERF_MAX_SAMPLE_STEPS_PER_RAY               256
#define NERF_CONST_POS_FREQS                        10
#define NERF_CONST_DIR_FREQS                        4
#define NERF_CONST_PTS_IN_DIM                       63
#define NERF_CONST_DIR_IN_DIM                       27
#define NERF_CONST_TRAIN_CHUNK_ROWS                 65536
#define NERF_CONST_SAMPLER_BLOCK_RAYS               128
#define NERF_CONST_SAMPLER_MAX_SAMPLE_STEPS_PER_RAY 256
#define NERF_CONST_DENSITY_INPUT_DIM                64
#define NERF_CONST_DENSITY_OUTPUT_DIM               16
#define NERF_CONST_DENSITY_WIDTH                    128
#define NERF_CONST_DENSITY_HIDDEN_LAYERS            5
#define NERF_CONST_GEO_FEATURE_DIM                  15
#define NERF_CONST_COLOR_INPUT_DIM                  48
#define NERF_CONST_COLOR_OUTPUT_DIM                 3
#define NERF_CONST_COLOR_OUTPUT_PADDED_DIM          16
#define NERF_CONST_COLOR_WIDTH                      128
#define NERF_CONST_COLOR_HIDDEN_LAYERS              3
#define NERF_CONST_NETWORK_BATCH_GRANULARITY        256
#define NERF_CONST_NETWORK_CHECKPOINT_MAX_TENSORS   32
#define NERF_CONST_FUSED_WIDTH                      128
#define NERF_CONST_FUSED_OUTPUT_WIDTH               16
#define NERF_CONST_FUSED_SKEW                       8
#define NERF_CONST_FUSED_INPUT_SKEW                 8
#define NERF_CONST_FUSED_BLOCK_ROWS                 8
#define NERF_CONST_FUSED_ITERS                      8
#define NERF_CONST_FUSED_BATCH_QUANTUM              128
#define NERF_CONST_FUSED_FORWARD_SHMEM_DENSITY      39168
#define NERF_CONST_FUSED_FORWARD_SHMEM_COLOR        39168
#define NERF_CONST_FUSED_BACKWARD_SHMEM             34816
#define NERF_CONST_FUSED_ELEMS_PER_LOAD             2048
#define NERF_CONST_FUSED_WEIGHTS_STRIDE             16384
#define NERF_CONST_WMMA_THREADS_X                   32
#define NERF_CONST_WMMA_THREADS_Y                   8
#define NERF_CONST_WMMA_THREADS_Z                   1
#define NERF_CONST_INPUT_GRAD_THREADS_X             16
#define NERF_CONST_INPUT_GRAD_THREADS_Y             16
#define NERF_CONST_INPUT_GRAD_THREADS_Z             1
#define NERF_CONST_THREADS_256                      256
#define NERF_CONST_ARENA_ALIGN_BYTES                16
#define NERF_CONST_OCCUPANCY_BLOCK_X                256
#define NERF_CONST_CONVERT_THREADS                  256
#define NERF_CONST_NETWORK_LOSS_SCALE               128.0f
#define NERF_CONST_SAMPLER_EPS                      1e-8f
#define NERF_CONST_INV255                           (1.0f / 255.0f)
#define NERF_CONST_BLAS_ALPHA                       1.0f
#define NERF_CONST_BLAS_BETA                        0.0f
#define NERF_CONST_GLOBAL_GRAD_CLIP_NORM            1.0f
#define NERF_CONST_UPDATE_GUARD_GRAD_NORM           100.0f
#define NERF_CONST_INV_LOSS_SCALE                   (1.0f / NERF_CONST_NETWORK_LOSS_SCALE)

typedef enum NerfStatus {
    NERF_STATUS_OK                            = 0,
    NERF_STATUS_INVALID_ARGUMENT              = 1,
    NERF_STATUS_OUT_OF_MEMORY                 = 2,
    NERF_STATUS_CUDA_FAILURE                  = 3,
    NERF_STATUS_DATASET_NOT_LOADED            = 4,
    NERF_STATUS_TRAINING_NOT_CONFIGURED       = 5,
    NERF_STATUS_RANGE_ERROR                   = 6,
    NERF_STATUS_OVERFLOW                      = 7,
    NERF_STATUS_INTERNAL_ERROR                = 8,
    NERF_STATUS_FILE_NOT_FOUND                = 9,
    NERF_STATUS_JSON_PARSE_FAILED             = 10,
    NERF_STATUS_MISSING_REQUIRED_FIELD        = 11,
    NERF_STATUS_INVALID_TRANSFORM_MATRIX      = 12,
    NERF_STATUS_IMAGE_LOAD_FAILED             = 13,
    NERF_STATUS_INCONSISTENT_IMAGE_RESOLUTION = 14,
    NERF_STATUS_CONFIGURATION_MISMATCH        = 15,
    NERF_STATUS_CHECKPOINT_INVALID            = 16,
    NERF_STATUS_CHECKPOINT_MISMATCH           = 17
} NerfStatus;

typedef struct NerfCreateDesc {
    uint32_t occupancy_grid_res;
    uint32_t max_sample_steps;
    uint32_t max_batch_rays;
    uint64_t arena_alignment_bytes;
} NerfCreateDesc;
NERF_API NerfStatus nerf_create_context(const NerfCreateDesc* desc, void** out_context);
NERF_API NerfStatus nerf_destroy_context(void* context);

typedef struct NerfVec3 {
    float x;
    float y;
    float z;
} NerfVec3;
typedef enum NerfCoordAxis { NERF_COORD_AXIS_POSITIVE_X = 0, NERF_COORD_AXIS_NEGATIVE_X = 1, NERF_COORD_AXIS_POSITIVE_Y = 2, NERF_COORD_AXIS_NEGATIVE_Y = 3, NERF_COORD_AXIS_POSITIVE_Z = 4, NERF_COORD_AXIS_NEGATIVE_Z = 5 } NerfCoordAxis;
typedef struct NerfCoordBasis {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} NerfCoordBasis;
typedef struct NerfCoordSystem {
    NerfCoordBasis world;
    NerfCoordBasis camera;
} NerfCoordSystem;
typedef struct NerfDatasetLoadDesc {
    const char* path_utf8;
    NerfVec3 offset;
    float scale;
    NerfCoordSystem source_system;
    NerfCoordSystem target_system;
} NerfDatasetLoadDesc;
typedef struct NerfDatasetInfo {
    uint32_t image_count;
    uint32_t image_width;
    uint32_t image_height;
    uint64_t images_bytes;
    uint64_t c2w_bytes;
    float fx;
    float fy;
    float cx;
    float cy;
    NerfCoordSystem source_system;
    NerfCoordSystem target_system;
} NerfDatasetInfo;
NERF_API NerfStatus nerf_load_dataset(void* context, const NerfDatasetLoadDesc* desc, NerfDatasetInfo* out_info);

typedef struct NerfHyperParams {
    float learning_rate;
    float adam_beta1;
    float adam_beta2;
    float adam_eps;
    uint32_t lr_decay_ksteps;
} NerfHyperParams;
typedef struct NerfOccupancyParams {
    float decay;
    float threshold;
    uint32_t cells_per_update;
    uint32_t update_interval;
    uint32_t warmup_steps;
} NerfOccupancyParams;
typedef enum NerfTrainCameraMode { NERF_TRAIN_CAMERA_MODE_FIXED = 0, NERF_TRAIN_CAMERA_MODE_CYCLE = 1, NERF_TRAIN_CAMERA_MODE_RANDOM = 2 } NerfTrainCameraMode;
typedef struct NerfTrainingConfig {
    NerfVec3 aabb_min;
    NerfVec3 aabb_max;
    NerfHyperParams hyper_params;
    NerfOccupancyParams occupancy_params;
    uint32_t rays_per_batch;
    uint32_t max_sample_steps_per_ray;
    uint32_t train_camera_mode;
    uint32_t fixed_train_camera_idx;
} NerfTrainingConfig;
NERF_API NerfStatus nerf_configure_training(void* context, const NerfTrainingConfig* config);

typedef struct NerfStepRequest {
    uint32_t rays_per_batch;
    uint32_t max_sample_steps_per_ray;
} NerfStepRequest;
NERF_API NerfStatus nerf_train_step(void* context, const NerfStepRequest* request);

typedef struct NerfTrainStats {
    float loss;
    float grad_norm;
    uint32_t has_nonfinite;
    uint32_t completed_steps;
    uint32_t last_train_camera;
} NerfTrainStats;
NERF_API NerfStatus nerf_read_train_stats(void* context, NerfTrainStats* out_stats);

typedef struct NerfCheckpointFileDesc {
    const char* path_utf8;
} NerfCheckpointFileDesc;
NERF_API NerfStatus nerf_save_network_weights(void* context, const NerfCheckpointFileDesc* desc);
NERF_API NerfStatus nerf_load_network_weights(void* context, const NerfCheckpointFileDesc* desc);

#ifdef __cplusplus
}
#endif

#endif
