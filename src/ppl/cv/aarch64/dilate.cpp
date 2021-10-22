// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for subitional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/cv/aarch64/dilate.h"
#include "ppl/common/log.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "morph.hpp"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <float.h>

namespace ppl {
namespace cv {
namespace aarch64 {

::ppl::common::RetCode armmaxFilter_f(int height, int width, int inWidthStride,
    const float* inData, int kernelx_len, int kernely_len,
    int outWidthStride, float* outData, int cn, float border_value){
    float minimal = -FLT_MAX;

    float *gRowMax = (float*) malloc(height * inWidthStride * sizeof(float));
    int leftPad = cn * (kernely_len >> 1);
    int rightPad = cn * width - leftPad;
    if (!(kernely_len & 1)) rightPad += cn;

    for (int i = 0; i < height; ++i){
        int inIndex = i * inWidthStride;

        for (int j = 0; j < leftPad; ++j){
            int yEnd = j - leftPad + cn * kernely_len;
            float _max = border_value;
            for (int jj = j % cn; jj < yEnd; jj += cn)
                if (inData[inIndex + jj] > _max) _max = inData[inIndex + jj];
            gRowMax[inIndex + j] = _max;
        }

        int j;
        for (j = leftPad; j < rightPad - 4; j += 4){
            float32x4_t mm_max = vdupq_n_f32(0);
            for (int jj = j - leftPad; jj < j - leftPad + cn * kernely_len; jj += cn){
                float32x4_t mm_temp = vld1q_f32(inData + inIndex + jj);
                mm_max = vmaxq_f32(mm_max, mm_temp);
            }
            vst1q_f32(gRowMax + inIndex + j, mm_max);
        }
        for (; j < width * cn; ++j){
            int yStart = j - leftPad;
            float _max = (j < rightPad)? minimal : border_value;
            int yEnd = yStart + cn * kernely_len;
            yEnd = std::min<int>(yEnd, width * cn);
            for (int jj = yStart; jj < yEnd; jj += cn)
                if (inData[inIndex + jj] > _max) _max = inData[inIndex + jj];
            gRowMax[inIndex + j] = _max;
        }
    }

    int upPad = kernelx_len >> 1;
    int downPad = height - upPad;
    if (!(kernelx_len & 1)) ++downPad;

    for (int i = 0; i < height; ++i){
        int xStart = i - upPad;
        int xEnd = xStart + kernelx_len;
        bool valid = (xStart >= 0) && (xEnd <= height);
        xEnd = std::min<int>(xEnd, height);
        xStart = std::max<int>(xStart, 0);
        int j = 0;
        for (; j < width * cn - 4; j += 4){
            float32x4_t mm_max = vdupq_n_f32(valid? minimal : border_value);
            for (int ii = xStart; ii < xEnd; ++ii){
                float32x4_t mm_temp = vld1q_f32(gRowMax + ii * inWidthStride + j);
                mm_max = vmaxq_f32(mm_temp, mm_max);
            }
            vst1q_f32(outData + i * outWidthStride + j, mm_max);
        }
        for (; j < width * cn; ++j){
            float _max = valid? minimal : border_value;
            for (int ii = xStart; ii < xEnd; ++ii){
                if (gRowMax[ii * inWidthStride + j] > _max) _max = gRowMax[ii * inWidthStride + j];
            }
            outData[i * outWidthStride + j] = _max;
        }
    }

    free(gRowMax);
    return ppl::common::RC_SUCCESS;
}

template<typename T>
::ppl::common::RetCode armmaxFilter_normal(int height, int width, int inWidthStride, 
    const T* inData, int kernelx_len, int kernely_len, const uchar* kernel, 
    int outWidthStride, T* outData, int cn, T border_value){
    T minimal;
    if (std::is_same<T, float>::value) {
        minimal = -FLT_MAX;
    } else if (std::is_same<T, uchar>::value) {
        minimal = 0;
    }
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < cn; ++c) {
                T _max = minimal;
                for (int ky = 0; ky < kernely_len; ++ky) {
                    int src_y = i + ky - (kernely_len >> 1);
                    bool valid_y = ((src_y >= 0) && (src_y < height));
                    for (int kx = 0; kx < kernelx_len; ++kx) {
                        int src_x = j + kx - (kernelx_len >> 1);
                        bool valid_x = ((src_x >= 0) && (src_x < width));
                        if (kernel[ky * kernelx_len + kx]) {
                            T value =  (valid_x && valid_y)? inData[src_y * inWidthStride + src_x * cn + c] : border_value;
                            _max = std::max(_max, value);
                        }
                    }
                }
                outData[i * outWidthStride + j * cn + c] = _max;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template<>
::ppl::common::RetCode Dilate<float, 1>(int height, int width, int inWidthStride,
    const float* inData, int kernelx_len, int kernely_len, const uchar* kernel,
    int outWidthStride, float* outData, BorderType border_type, float border_value){
    assert(inData != NULL);
    assert(outData != NULL);
    assert(kernel != NULL);
    assert(height != 0 && width != 0 && inWidthStride != 0 && outWidthStride != 0);
    if (border_type != BORDER_TYPE_CONSTANT) {
        border_value = std::numeric_limits<float>::lowest();
    }
    bool flag = true;
    for (int i = 0; i < kernelx_len * kernely_len; ++i){
        if (kernel[i] != 1){
            flag = false;
            break;
        }
    }
    // if (flag) return armmaxFilter_f(height, width, inWidthStride, inData, kernelx_len, kernely_len,
    //     outWidthStride, outData, 1, border_value);
    if (flag) ppl::cv::aarch64::morph_f32<DilateVecOp, 1, 3>(height, width, inWidthStride, inData, 
                                                                    outWidthStride, outData, ppl::cv::BORDER_TYPE_CONSTANT, border_value);
    else return armmaxFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len,
        kernel, outWidthStride, outData, 1, border_value);
}

template<>
::ppl::common::RetCode Dilate<float, 3>(int height, int width, int inWidthStride,
    const float* inData, int kernelx_len, int kernely_len, const uchar* kernel,
    int outWidthStride, float* outData, BorderType border_type, float border_value){
    assert(inData != NULL);
    assert(outData != NULL);
    assert(kernel != NULL);
    assert(height != 0 && width != 0 && inWidthStride != 0 && outWidthStride != 0);
    if (border_type != BORDER_TYPE_CONSTANT) {
        border_value = std::numeric_limits<float>::lowest();
    }
    bool flag = true;
    for (int i = 0; i < kernelx_len * kernely_len; ++i){
        if (kernel[i] != 1){
            flag = false;
            break;
        }
    }
    // if (flag) return armmaxFilter_f(height, width, inWidthStride, inData, kernelx_len, kernely_len,
    //     outWidthStride, outData, 3, border_value);
    if (flag) ppl::cv::aarch64::morph_f32<DilateVecOp, 3, 3>(height, width, inWidthStride, inData, 
                                                                    outWidthStride, outData, ppl::cv::BORDER_TYPE_CONSTANT, border_value);
    else return armmaxFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len,
        kernel, outWidthStride, outData, 3, border_value);
}

template<>
::ppl::common::RetCode Dilate<float, 4>(int height, int width, int inWidthStride,
    const float* inData, int kernelx_len, int kernely_len, const uchar* kernel,
    int outWidthStride, float* outData, BorderType border_type, float border_value){
    assert(inData != NULL);
    assert(outData != NULL);
    assert(kernel != NULL);
    assert(height != 0 && width != 0 && inWidthStride != 0 && outWidthStride != 0);
    if (border_type != BORDER_TYPE_CONSTANT) {
        border_value = std::numeric_limits<float>::lowest();
    }
    bool flag = true;
    for (int i = 0; i < kernelx_len * kernely_len; ++i){
        if (kernel[i] != 1){
            flag = false;
            break;
        }
    }
    // if (flag) return armmaxFilter_f(height, width, inWidthStride, inData, kernelx_len, kernely_len,
    //     outWidthStride, outData, 4, border_value);
    if (flag) ppl::cv::aarch64::morph_f32<DilateVecOp, 4, 3>(height, width, inWidthStride, inData, 
                                                                    outWidthStride, outData, ppl::cv::BORDER_TYPE_CONSTANT, border_value);
    else return armmaxFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len,
        kernel, outWidthStride, outData, 4, border_value);
}


}
}
} // namespace ppl::cv::aarch64
