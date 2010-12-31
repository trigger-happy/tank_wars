#ifndef OPERATIONS_H
#define OPERATIONS_H
#include <cuda.h>

#ifdef __CUDA_ARCH__

#define ADD(x,y) __fadd_rn(x,y)
#define MUL(x,y) __fmul_rn(x,y)

#else

#define ADD(x,y) (x + y)
#define MUL(x,y) (x * y)

#endif

#endif //OPERATIONS_H