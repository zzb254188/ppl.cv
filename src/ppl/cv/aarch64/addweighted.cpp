// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
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

#include "ppl/cv/aarch64/addweighted.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <arm_neon.h>
#include <algorithm>
namespace ppl {
namespace cv {
namespace aarch64 {


::ppl::common::RetCode addWeighted_f32(
    int height,
    int width,
    int channels,
    int inWidthStride0,
    const float *inData0,
    float alpha,
    int inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int outWidthStride,
    float *outData) {

    width *= channels;
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta = vdupq_n_f32(beta);
    float32x4_t vgamma = vdupq_n_f32(gamma);

    for (int i = 0; i < height; ++i) {
        int j = 0;
        for (; j <= width - 4; j += 4) {
            float32x4_t vdata0 = vld1q_f32(inData0 + i * inWidthStride0 + j);
            float32x4_t vdata1 = vld1q_f32(inData1 + i * inWidthStride1 + j);
            float32x4_t vdst = vgamma;
            vdst = vmlaq_f32(vdst, vdata0, valpha);
            vdst = vmlaq_f32(vdst, vdata1, vbeta);
            vst1q_f32(outData + i * outWidthStride + j, vdst);
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] =
                inData0[i * inWidthStride0 + j] * alpha +
                inData1[i * inWidthStride1 + j] * beta +
                gamma;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template<>
::ppl::common::RetCode AddWeighted<float, 1>(int height, int width, int inWidthStride0, const float *inData0,
    float alpha, int inWidthStride1, const float *inData1, float beta, float gamma,
    int outWidthStride, float *outData) {
    return addWeighted_f32(height, width, 1, inWidthStride0, inData0, alpha, inWidthStride1,
        inData1, beta, gamma, outWidthStride, outData);
}

template<>
::ppl::common::RetCode AddWeighted<float, 3>(int height, int width, int inWidthStride0, const float *inData0,
    float alpha, int inWidthStride1, const float *inData1, float beta, float gamma,
    int outWidthStride, float *outData) {
    return addWeighted_f32(height, width, 3, inWidthStride0, inData0, alpha, inWidthStride1,
        inData1, beta, gamma, outWidthStride, outData);
}

template<>
::ppl::common::RetCode AddWeighted<float, 4>(int height, int width, int inWidthStride0, const float *inData0,
    float alpha, int inWidthStride1, const float *inData1, float beta, float gamma,
    int outWidthStride, float *outData) {
    return addWeighted_f32(height, width, 4, inWidthStride0, inData0, alpha, inWidthStride1,
        inData1, beta, gamma, outWidthStride, outData);
}

}//! namespace aarch64
}//! namespace cv
}//! namespace ppl

