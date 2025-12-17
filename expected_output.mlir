// Expected output after Option C: Custom Phased Bufferization Strategy
// Phase 1: Function signatures converted to memref
// Phase 2: Bufferization applied
// Phase 3: to_tensor/to_buffer artifacts eliminated

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>

module {
  // Function signature converted to memref in Phase 1
  func.func @main(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) -> memref<2x3xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x3xf32>, memref<2x3xf32>) outs(%alloc : memref<2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    // No to_tensor needed - return memref directly (Phase 3 cleanup)
    return %alloc : memref<2x3xf32>
  }

  // Complex test function with memref signature
  func.func @complex_test(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<4xf32>, memref<4xf32>) outs(%alloc : memref<4xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_1 : f32
      linalg.yield %0 : f32
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc, %arg0 : memref<4xf32>, memref<4xf32>) outs(%alloc_0 : memref<4xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      linalg.yield %0 : f32
    }
    // No to_tensor needed - return memref directly
    return %alloc_0 : memref<4xf32>
  }
}