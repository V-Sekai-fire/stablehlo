// Simple StableHLO program for RISC-V codegen testing
func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}
