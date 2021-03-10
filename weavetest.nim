import typetraits
import sequtils
import sugar

import arraymancer
import fftw3
# import benchy
import timelog

import weave


#############################################################
## Coordinate Helpers
#############################################################
proc getCoord2(index: int, shape: openArray[int]): array[2, int] {.inline.} =
  var index = index
  result[0] = index div shape[1]
  index = index - result[0]*shape[1]
  result[1] = index

proc getCoord3(index: int, shape: openArray[int]): array[3, int] {.inline.} =
  var index = index
  result[0] = index div (shape[2]*shape[1])
  index = index - result[0]*shape[2]*shape[1]
  result[1] = index div shape[2]
  index = index - result[1]*shape[2]
  result[2] = index

proc getCoord(index: int, shape: seq[int]): seq[int] {.inline.} =
  if shape.len == 2:
    result = @(getCoord2(index, shape))
  elif shape.len == 3:
    result = @(getCoord3(index, shape))
  else:
    doAssert(false, "shape.len == 2 or 3")

#############################################################
## Tensor Helpers
#############################################################
type
  Metadata = tuple[offset, rank: int, strides, shape: seq[int]]

proc getMeta[T](t: Tensor[T]): Metadata =
  result.offset = t.offset
  result.rank = t.rank
  result.strides = t.strides.toSeq
  result.shape = t.shape.toSeq

#############################################################
## Index Helpers
#############################################################
proc getIndex*(m: Metadata, idx: varargs[int]): int {.noSideEffect, inline.} =
  result = m.offset
  for i in 0..<idx.len:
    result += m.strides[i]*idx[i]

proc getNewIndex*(m: Metadata, shifts: openArray[int], idx: varargs[int]): int {.noSideEffect, inline.} =
  result = m.offset
  for i in 0..<idx.len:
    let newidx = (idx[i] + shifts[i]) mod m.shape[i]
    result += m.strides[i]*newidx

# proc newidx(idx, shifts, shapes: openArray[int]): seq[int] {.inline.} =
#   let I = idx.len()
#   doAssert I <= 3
#   result = newSeq[int](I)
#   for i in 0..<I:
#     result[i] = (idx[i] + shifts[i]) mod shapes[i]

#############################################################
## Weave
#############################################################
proc circshift2_weave[T](inBuf, outBuf: ptr UncheckedArray[T], m: Metadata, shifts: seq[int]) =
  parallelFor i in 0..<m.shape[0]:
    captures: {inBuf, outBuf, m, shifts}
    parallelFor j in 0..<m.shape[1]:
      captures: {inBuf, outBuf, m, shifts, i}
      outBuf[getNewIndex(m, shifts, i, j)] = inBuf[getIndex(m, i, j)]

proc circshift3_weave[T](inBuf, outBuf: ptr UncheckedArray[T], m: Metadata, shifts: seq[int]) =
  parallelFor i in 0..<m.shape[0]:
    captures: {inBuf, outBuf, m, shifts}
    parallelFor j in 0..<m.shape[1]:
      captures: {inBuf, outBuf, m, shifts, i}
      parallelFor k in 0..<m.shape[2]:
        captures: {inBuf, outBuf, m, shifts, i, j}
        outBuf[getNewIndex(m, shifts, i, j, k)] = inBuf[getIndex(m, i, j, k)]

proc circshift_weave[T](inBuf, outBuf: ptr UncheckedArray[T], m: Metadata, shifts: seq[int]) =
  init(Weave)
  if shifts.len == 2:
    circshift2_weave(inBuf, outBuf, m, shifts)
  elif shifts.len == 3:
    circshift3_weave(inBuf, outBuf, m, shifts)
  exit(Weave)


#############################################################
## OpenMp
#############################################################
proc circshift_openmp[T](inBuf, outBuf: ptr UncheckedArray[T], m: Metadata, shifts: seq[int]) =
  let iterableSize = m.shape[0]*m.shape[1] - 1
  for idx in 0||iterableSize:
    let idx = getCoord(idx, m.shape)
    outBuf[getNewIndex(m, shifts, idx)] = inBuf[getIndex(m, idx)]

#############################################################
## Benchy
#############################################################
type
  ParallelMethod = enum
    parWeave, parOpenMp

proc fftshift_parallel[T](t: Tensor[T], parmethod: ParallelMethod): Tensor[T] =
  let
    shape = t.shape.toSeq
    shifts = t.shape.toSeq.map(x => x div 2)
  # Alloc Tensor
  result = newTensor[T](shape)

  let
    ptrIn = t.unsafe_raw_offset().distinctBase()
    ptrOut = result.unsafe_raw_offset().distinctBase()

  case parmethod
  of parWeave:
    # init(Weave)
    circshift_weave[T](ptrIn, ptrOut, getMeta(t), shifts)
    # exit(Weave)
  of parOpenMp:
    circshift_openmp[T](ptrIn, ptrOut, getMeta(t), shifts)

proc testMe(DIMS: openArray[int]) =
  echo "RUNNING BENCH ON >>", DIMS
  var tensor = randomTensor[float](DIMS, 100.0)
  timeIt("current_FftShift"):
    var refOutput = fftshift(tensor)

  block:
    timeIt("weave_FftShift"):
      var weaveOutput = fftshift_parallel(tensor, ParallelMethod.parWeave)
    echo mean_relative_error(weaveOutput, refOutput)

  block:
    timeIt("openMp_FftShift"):
      var openMpOutput = fftshift_parallel(tensor, ParallelMethod.parOpenMp)
    echo mean_relative_error(openMpOutput, refOutput)

proc main() =
  # init(Weave)
  testMe([100, 100])
  testMe([20, 30, 40])
  # exit(Weave)

when isMainModule:
  main()
