import typetraits
import sequtils
import sugar

import arraymancer
import fftw3

#############################################################
## Tensor Helpers
#############################################################
type
  TupShape = tuple[offset: int, strides, shape: seq[int]]

proc getMeta[T](t: Tensor[T]): TupShape =
  result.offset = t.offset
  result.strides = t.strides.toSeq
  result.shape = t.shape.toSeq

#############################################################
## Coordinate Helpers
#############################################################
func get2DCoord(index: int, shape: seq[int]): array[2, int] {.inline.} =
  var index = index
  result[0] = index div shape[1]
  index = index - result[0]*shape[1]
  result[1] = index

func get3DCoord(index: int, shape: seq[int]): array[3, int] {.inline.} =
  var index = index
  result[0] = index div (shape[2]*shape[1])
  index = index - result[0]*shape[2]*shape[1]
  result[1] = index div shape[2]
  index = index - result[1]*shape[2]
  result[2] = index

#############################################################
## OpenMp
#############################################################
template circshift_openmp_impl[T](inBuf, outBuf: ptr UncheckedArray[T], m: TupShape, shifts: seq[int], getCoord: untyped) =
  let iterableSize = foldl(m.shape, a*b) - 1
  for idx in 0||iterableSize:
    let indices = getCoord(idx, m.shape)
    outBuf[getShiftedIndex(m.offset, m.strides, m.shape, shifts, indices)] = inBuf[getIndex(m.offset, m.strides, m.shape, indices)]

proc circshift_openmp[T](inBuf, outBuf: ptr UncheckedArray[T], m: TupShape, shifts: seq[int]) =
  if shifts.len == 2:
    var getCoord = get2DCoord
    circshift_openmp_impl(inBuf, outBuf, m, shifts, getCoord)

  elif shifts.len == 3:
    var getCoord = get3DCoord
    circshift_openmp_impl(inBuf, outBuf, m, shifts, getCoord)

#############################################################
## Benchy
#############################################################
proc fftshift_openmp*[T](t: Tensor[T]): Tensor[T] =
  let
    shape = t.shape.toSeq
    shifts = t.shape.toSeq.map(x => x div 2)
  # Alloc Tensor
  result = newTensor[T](shape)
  let
    ptrIn = t.unsafe_raw_offset().distinctBase()
    ptrOut = result.unsafe_raw_offset().distinctBase()
  circshift_openmp[T](ptrIn, ptrOut, getMeta(t), shifts)
#
