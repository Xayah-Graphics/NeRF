#ifndef NERF_H
#define NERF_H

#ifdef _WIN32
#ifdef NERF_BUILD_SHARED
#define NERF_API __declspec(dllexport)
#else
#define NERF_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define NERF_API __attribute__((visibility("default")))
#else
#define NERF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // NERF_H
