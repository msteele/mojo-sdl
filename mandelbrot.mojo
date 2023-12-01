# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import benchmark
from complex import ComplexSIMD, ComplexFloat64
from math import iota, clamp, log, pow
from python import Python
from runtime.llcl import num_cores
from algorithm import parallelize, vectorize
from tensor import Tensor
from utils.index import Index

from memory.unsafe import bitcast
from SDL import *
from cmap import cmap

alias float_type = DType.float64
alias simd_width = 2 * simdwidthof[float_type]()

alias width = 480
alias height = 480
alias MAX_ITERS = 500

alias MIN_X = -2.0
alias MAX_X = 0.6
alias MIN_Y = -1.5
alias MAX_Y = 1.5

@no_inline
fn mandelbrot_kernel_SIMD[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[float_type, simd_width]:
    """A vectorized implementation of the inner mandelbrot computation."""
    let cx = c.re
    let cy = c.im
    var x = SIMD[float_type, simd_width](0)
    var y = SIMD[float_type, simd_width](0)
    var y2 = SIMD[float_type, simd_width](0)
    var iters = SIMD[float_type, simd_width](0)

    var t: SIMD[DType.bool, simd_width] = True
    for i in range(MAX_ITERS):
        if not t.reduce_or():
            break
        y2 = y * y
        y = x.fma(y + y, cy)
        t = x.fma(x, y2) <= 4
        x = x.fma(x, cx - y2)
        iters = t.select(iters + 1, iters)
    return iters


fn main() raises:

    let t = Tensor[float_type](height, width)

    var min_x: Float64 = MIN_X
    var max_x: Float64 = MAX_X
    var min_y: Float64 = MIN_Y
    var max_y: Float64 = MAX_Y

    @parameter
    @no_inline
    fn worker(row: Int):
        let scale_x = (max_x - min_x) / width
        let scale_y = (max_y - min_y) / height

        #print(scale_x)
        @parameter
        @no_inline
        fn compute_vector[simd_width: Int](col: Int):
            """Each time we operate on a `simd_width` vector of pixels."""
            let cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            let cy = min_y + row * scale_y
            let c = ComplexSIMD[float_type, simd_width](cx, cy)
            t.data().simd_store[simd_width](
                row * width + col, mandelbrot_kernel_SIMD[simd_width](c)
            )

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[simd_width, compute_vector](width)

    _ = t  # Make sure tensor isn't destroyed before benchmark is finished

    var sdl = SDL()

    let res = sdl.Init(0x00000020)
    let window = sdl.CreateWindow(StringRef("Mandelbrot").data,
                                  SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                  width, height, SDL_WINDOW_SHOWN)

    let renderer = sdl.CreateRenderer(window, -1, 0)

    let display = sdl.CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, width, height)

    _ = sdl.SetRenderTarget(renderer, display)

    fn redraw(sdl: SDL, t: Tensor[float_type]) raises:
        let gamma: Float64 = 0.3
        let cpow_max = pow(Float64(MAX_ITERS),gamma)
        _ = sdl.SetRenderTarget(renderer, display)
        for y in range(t.shape()[1]):
            for x in range(t.shape()[0]):
                let val = t[x,y]
                let cpow = pow(val, gamma)
                let color = cmap[(255.0*cpow/cpow_max).to_int()]
                let r = color[0]
                let g = color[1]
                let b = color[2]
                _ = sdl.SetRenderDrawColor(renderer, r, g, b, 255)
                _ = sdl.RenderDrawPoint(renderer, y, x)

        _ = sdl.SetRenderTarget(renderer, 0)
        _ = sdl.RenderCopy(renderer, display, 0, 0)
        _ = sdl.RenderPresent(renderer)

    var event = Event()


    fn screen_to_world(sx: Int32, sy: Int32, inout wx: Float64, inout wy: Float64):
        let fsx = sx.cast[DType.float64]()
        let fsy = sy.cast[DType.float64]()
        wx = (max_x - min_x) * fsx/Float64(width) + min_x
        wy = (max_y - min_y) * fsy/Float64(height) + min_y

    var running = True
    var dirty = True

    while running:

        while sdl.PollEvent(Pointer[Event].address_of(event)) != 0:
            if (event.type == SDL_MOUSEWHEEL):
                let mwe = event.as_mousewheel()
                #print(mwe.preciseX, mwe.preciseY)
                let scale = (1 + mwe.preciseY.cast[DType.float64]() / 20.0)
                min_x = (min_x * scale)
                max_x = (max_x * scale)
                min_y = (min_y * scale)
                max_y = (max_y * scale)
                dirty = True
            if (event.type == SDL_QUIT):
                running = False
                break

        if dirty:
            parallelize[worker](height, height)

            dirty = False
        redraw(sdl, t)

        _= sdl.Delay((1000 / 120).to_int())

    _ = t
    sdl.DestroyWindow(window)
    sdl.Quit()
