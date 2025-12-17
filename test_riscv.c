// Generated C99 code from StableHLO
// Compile with: gcc -std=c99 ...

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

void main(float* restrict arg0, int64_t arg0_shape[2], float* restrict arg1, int64_t arg1_shape[2], float* restrict result, int64_t result_shape[2]) {
  int64_t i0, i1;
for (i0 = 0; i0 < result_shape[0]; i0++) {
    for (i1 = 0; i1 < result_shape[1]; i1++) {
      result[i0 * result_shape[1] + i1] = arg0[i0 * arg0_shape[1] + i1] + arg1[i0 * arg1_shape[1] + i1];
    }
  }

}

void complex_test(float* restrict arg0, int64_t arg0_shape[1], float* restrict arg1, int64_t arg1_shape[1], float* restrict result, int64_t result_shape[1]) {
  // Allocate temp0
  int64_t temp0_size = 1 * 4;
  float* temp0 = (float*)malloc(temp0_size * sizeof(float));
  int64_t temp0_shape[1] = {4};

  int64_t i0;
for (i0 = 0; i0 < temp0_shape[0]; i0++) {
    temp0[i0] = arg0[i0] * arg1[i0];
  }

  int64_t i0;
for (i0 = 0; i0 < result_shape[0]; i0++) {
    result[i0] = temp0[i0] + arg0[i0];
  }

  free(temp0);
}

