import typetraits
import sequtils

import arraymancer
import fftw3
import benchy

import fftshift_openmp
proc testMe(DIMS: openArray[int]) =
  echo "RUNNING BENCH ON >>", DIMS
  var tensor = randomTensor[float](DIMS, 100.0)

  var refOutput : Tensor[float]
  timeIt("current_FftShift"):
    refOutput = fftshift(tensor)
  keep(refOutput)

  var weaveOutput : Tensor[float]
  timeIt("weave_FftShift"):
    weaveOutput = fftshift_parallel(tensor)
  keep(weaveOutput)
  echo mean_relative_error(weaveOutput, refOutput)

  var openMpOutput : Tensor[float]
  timeIt("openMp_FftShift"):
    openMpOutput = fftshift_openmp(tensor)
  keep(openMpOutput)
  echo mean_relative_error(openMpOutput, refOutput)

proc main() =
  testMe([1000, 1000])
  testMe([200, 300, 400])

when isMainModule:
  main()
