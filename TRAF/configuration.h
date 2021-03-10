#ifndef EXAMPLE_TRAFFIC_CONFIGURATION_H
#define EXAMPLE_TRAFFIC_CONFIGURATION_H
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
static const int kSeed = 42;
static const int kMaxVelocity = 10;
static const int kMaxDegree = 4;

#include <ctime>
using namespace std::chrono;
#ifndef PARAM_SIZE
static const int kNumIntersections = 850;
#else
static const int kNumIntersections = PARAM_SIZE;
#endif  // PARAM_SIZE

static const float kCellLength = 0.005f;
static const float kProducerRatio = 0.02f;
static const float kTargetRatio = 0.003f;
// #define PARAM_NUM_ITER 1
#ifdef PARAM_NUM_ITER
static const int kNumIterations = PARAM_NUM_ITER;
#else
static const int kNumIterations = 5000;
#endif  // PARAM_NUM_ITER
static const float kCarAllocationRatio = 0.02f;

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 2*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

// Only for baseline version.
static const int kMaxNumCells = kNumObjects;
static const int kMaxNumCars = kNumObjects;

#endif  // EXAMPLE_TRAFFIC_CONFIGURATION_H
