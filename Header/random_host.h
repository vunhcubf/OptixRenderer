#include "common.h"
#include "curand.h"
#include "curand_kernel.h"
class RandomSequenceGenerator{
public:
    curandStateScrambledSobol64_t* RandomStates;
    curandDirectionVectors64_t* DeviceVectors;
    uint64* ScrambleConstant;
    uint* PixelOffset;
};