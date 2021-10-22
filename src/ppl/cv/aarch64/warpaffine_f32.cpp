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

#include "ppl/cv/aarch64/warpaffine.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/log.h"
#include "common.hpp"
#include "operation_utils.hpp"
#include <arm_neon.h>
#include <limits.h>
#include <algorithm>
#include <cmath>

namespace ppl {
namespace cv {
namespace aarch64 {

const int AB_BITS = 10;
const int AB_SCALE = 1 << AB_BITS;
const int INTER_BITS = 5;
const int INTER_TAB_SIZE = 1 << INTER_BITS;
const int INTER_REMAP_COEF_BITS = 15;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

template<typename T>
inline static short saturate_cast_short(T x) {
    return x > SHRT_MIN ? x < SHRT_MAX ? x : SHRT_MAX : SHRT_MIN;
}

template<typename T>
inline T clip(T value, T min_value, T max_value) {
    return min(max(value, min_value), max_value);
}

template<typename T>
::ppl::common::RetCode warpAffine_nearest(
    T *dst, const T *src, 
    int inHeight, int inWidth, int inWidthStride, 
    int outHeight, int outWidth, int outWidthStride, 
    const float *M, int cn, int borderMode, float borderValue = 0.0f)
{
    int round_delta = AB_SCALE >> 1;
    int *adelta = (int *) malloc(outWidth * sizeof(int));
    int *bdelta = (int *) malloc(outWidth * sizeof(int));
    for (int x = 0; x < outWidth; ++x) {
        adelta[x] = rint(M[0] * x * AB_SCALE);
        bdelta[x] = rint(M[3] * x * AB_SCALE);
    }
    for (int i = 0; i < outHeight; ++i) {
        int X0 = rint((M[1] * i + M[2]) * AB_SCALE) + round_delta;
        int Y0 = rint((M[4] * i + M[5]) * AB_SCALE) + round_delta;
        prefetch(src + ((Y0 + bdelta[0]) >> AB_BITS) * inWidthStride + ((X0 + adelta[0]) >> AB_BITS) * cn, outWidth);
        for (int j = 0; j < outWidth; ++j) {
            int srcX = (X0 + adelta[j]) >> AB_BITS;
            int srcY = (Y0 + bdelta[j]) >> AB_BITS;
            int dstIndex = i * outWidthStride + j * cn;
            int srcIndex = srcY * inWidthStride + srcX * cn;
            // LOG(INFO) << srcY << " " << inWidthStride << " " << srcX << " " <<cn;
            // LOG(INFO) << srcIndex;
            if(borderMode == BORDER_TYPE_CONSTANT){
                if (srcX >= 0 && srcX < inWidth && srcY >= 0 && srcY < inHeight) {
                    for (int k = 0; k < cn; ++k) {
                        dst[dstIndex + k] = src[srcIndex + k];
                    }
                } else {
                    for (int k = 0; k < cn; ++k) {
                        dst[dstIndex + k] = borderValue;
                    }
                }
            }
            else if(borderMode == BORDER_TYPE_TRANSPARENT){
                if (srcX >= 0 && srcX < inWidth && srcY >= 0 && srcY < inHeight) { 
                    for (int k = 0; k < cn; ++k) {
                        dst[dstIndex + k] = src[srcIndex + k];
                    }
                } else {
                    continue;
                }
            }
            else if(borderMode == BORDER_TYPE_REPLICATE)
            {
                srcX = clip(srcX, 0, inWidth - 1);
                srcY = clip(srcY, 0, inHeight - 1);
                for (int k = 0; k < cn; ++k) {
                    dst[dstIndex + k] = src[srcIndex + k];
                }
            }
        }
    }
    free(adelta);
    free(bdelta);
}

static void initTab_linear_short(short *short_tab) {
    float scale = 1.f / INTER_TAB_SIZE;
    for (int i = 0; i < INTER_TAB_SIZE; ++i) {
        float vy = i * scale;
        for (int j = 0; j < INTER_TAB_SIZE; ++j, short_tab += 4) {
            float vx = j * scale;
            short_tab[0] = saturate_cast_short((1 - vy) * (1 - vx) * INTER_REMAP_COEF_SCALE);
            short_tab[1] = saturate_cast_short((1 - vy) * vx * INTER_REMAP_COEF_SCALE);
            short_tab[2] = saturate_cast_short(vy * (1 - vx) * INTER_REMAP_COEF_SCALE);
            short_tab[3] = saturate_cast_short(vy * vx * INTER_REMAP_COEF_SCALE);
        }
    }
}

::ppl::common::RetCode warpAffine_linear_float(float *dst, const float *src, 
    int inHeight, int inWidth, int inWidthStride, 
    int outHeight, int outWidth, int outWidthStride, 
    const float *M, int cn, int borderMode, float borderValue = 0.0f) {

    short *short_tab = (short *) malloc(INTER_TAB_SIZE * INTER_TAB_SIZE * 8 * sizeof(short));
    initTab_linear_short(short_tab);

    int round_delta = AB_SCALE / INTER_TAB_SIZE / 2;

    int *adelta = (int *) malloc(outWidth * sizeof(int));
    int *bdelta = (int *) malloc(outWidth * sizeof(int));
    for (int i = 0; i < outWidth; ++i) {
        adelta[i] = rint(M[0] * i * AB_SCALE);
        bdelta[i] = rint(M[3] * i * AB_SCALE);
    }
    const int BLOCK_SZ = 64;
    short XY_INT[BLOCK_SZ * BLOCK_SZ * 2], XY_DEC[BLOCK_SZ * BLOCK_SZ];
    int bh0 = std::min<int>(BLOCK_SZ / 2, outHeight);
    int bw0 = std::min<int>(BLOCK_SZ * BLOCK_SZ / bh0, outWidth);
    bh0 = std::min<int>(BLOCK_SZ * BLOCK_SZ / bw0, outHeight);

    for(int y = 0; y < outHeight; y += bh0) {
        int bh = std::min<int>(bh0, outHeight - y);
        for (int x = 0; x < outWidth; x += bw0) {
            int bw = std::min<int>(bw0, outWidth - x);
            for (int y1 = 0; y1 < bh; ++y1) {
                short *xy_int_p = XY_INT + y1 * bw * 2;
                short *xy_dec_p = XY_DEC + y1 * bw;
                int x_int = (int) ((M[1] * (y + y1) + M[2]) * AB_SCALE) + round_delta;
                int y_int = (int) ((M[4] * (y + y1) + M[5]) * AB_SCALE) + round_delta;

                int32x4_t dec_mask = vdupq_n_s32(INTER_TAB_SIZE - 1);
                int32x4_t m_X_int = vdupq_n_s32(x_int);
                int32x4_t m_Y_int = vdupq_n_s32(y_int);
                int x1 = 0;
                for (; x1 <= bw - 8; x1 += 8) {
                    int32x4_t tx0, tx1, ty0, ty1;
                    tx0 = vaddq_s32(m_X_int, vld1q_s32(adelta + x + x1));
                    tx1 = vaddq_s32(m_X_int, vld1q_s32(adelta + x + x1 + 4));
                    ty0 = vaddq_s32(m_Y_int, vld1q_s32(bdelta + x + x1));
                    ty1 = vaddq_s32(m_Y_int, vld1q_s32(bdelta + x + x1 + 4));

                    tx0 = vshrq_n_s32(tx0, AB_BITS - INTER_BITS);
                    tx1 = vshrq_n_s32(tx1, AB_BITS - INTER_BITS);
                    ty0 = vshrq_n_s32(ty0, AB_BITS - INTER_BITS);
                    ty1 = vshrq_n_s32(ty1, AB_BITS - INTER_BITS);

                    int16x8_t fx, fy;
                    fx = vcombine_s16(
                        vqmovn_s32(vandq_s32(tx0, dec_mask)),
                        vqmovn_s32(vandq_s32(tx1, dec_mask))
                        );
                    fy = vcombine_s16(
                        vqmovn_s32(vandq_s32(ty0, dec_mask)),
                        vqmovn_s32(vandq_s32(ty1, dec_mask))
                        );
                    int16x8_t final_f = vaddq_s16(fx, vshlq_n_s16(fy, INTER_BITS));

                    tx0 = vshrq_n_s32(tx0, INTER_BITS);
                    tx1 = vshrq_n_s32(tx1, INTER_BITS);
                    ty0 = vshrq_n_s32(ty0, INTER_BITS);
                    ty1 = vshrq_n_s32(ty1, INTER_BITS);
                    int16x4x2_t zip0;
                    zip0.val[0] = vqmovn_s32(tx0);
                    zip0.val[1] = vqmovn_s32(ty0);
                    int16x4x2_t zip1;
                    zip1.val[0] = vqmovn_s32(tx1);
                    zip1.val[1] = vqmovn_s32(ty1);

                    vst2_s16(xy_int_p + x1 * 2, zip0);
                    vst2_s16(xy_int_p + x1 * 2 + 8, zip1);
                    vst1q_s16(xy_dec_p + x1, final_f);
                }
                for (; x1 < bw; ++x1) {
                    int x_value = (x_int + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
                    int y_value = (y_int + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
                    xy_int_p[x1 * 2] = saturate_cast_short(x_value >> INTER_BITS);
                    xy_int_p[x1 * 2 + 1] = saturate_cast_short(y_value >> INTER_BITS);
                    xy_dec_p[x1] = (short)((y_value & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + 
                        (x_value & (INTER_TAB_SIZE - 1)));
                }
            }
            for (int y1 = 0; y1 < bh; ++y1) {
                int dstY = y1 + y;
                for (int x1 = 0; x1 < bw; ++x1) {
                    int srcX = XY_INT[2 * (y1 * bw + x1)];
                    int srcY = XY_INT[2 * (y1 * bw + x1) + 1];
                    int srcIndex = srcY * inWidthStride + srcX * cn;
                    int dstX = x1 + x;
                    int dstIndex = dstY * outWidthStride + dstX * cn;

                    bool flag[4];

                    if(borderMode == BORDER_TYPE_CONSTANT)
                    {
                        flag[0] = (srcX >= 0 && srcX < inWidth && srcY >= 0 && srcY < inHeight);
                        flag[1] = (srcX+1 >= 0 && srcX+1 < inWidth && srcY >= 0 && srcY < inHeight);
                        flag[2] = (srcX >= 0 && srcX < inWidth && srcY+1 >= 0 && srcY+1 < inHeight);
                        flag[3] = (srcX+1 >= 0 && srcX+1 < inWidth && srcY+1 >= 0 && srcY+1 < inHeight);
                        const short *p_short_tab = short_tab + XY_DEC[y1 * bw + x1] * 4;
                        float src_value[4];
                        for (int k = 0; k < cn; ++k) {
                            src_value[0] = flag[0] ? src[srcIndex + k] : borderValue;
                            src_value[1] = flag[1] ? src[srcIndex + cn + k] : borderValue;
                            src_value[2] = flag[2] ? src[srcIndex + inWidthStride + k] : borderValue;
                            src_value[3] = flag[3] ? src[srcIndex + inWidthStride + cn + k] : borderValue;

                            float sum = 0;
                            //linear interpolation
                            sum += src_value[0] * p_short_tab[0];
                            sum += src_value[1] * p_short_tab[1];
                            sum += src_value[2] * p_short_tab[2];
                            sum += src_value[3] * p_short_tab[3];
                            dst[dstIndex + k] = sum / INTER_REMAP_COEF_SCALE;
                        } 
                    }
                    else if(borderMode == BORDER_TYPE_TRANSPARENT)
                    {
                        flag[0] = (srcX >= 0 && srcX < inWidth && srcY >= 0 && srcY < inHeight);
                        flag[1] = (srcX+1 >= 0 && srcX+1 < inWidth && srcY >= 0 && srcY < inHeight);
                        flag[2] = (srcX >= 0 && srcX < inWidth && srcY+1 >= 0 && srcY+1 < inHeight);
                        flag[3] = (srcX+1 >= 0 && srcX+1 < inWidth && srcY+1 >= 0 && srcY+1 < inHeight);
                        if(flag[0] && flag[1] && flag[2] && flag[3])
                        {
                            const short *p_short_tab = short_tab + XY_DEC[y1 * bw + x1] * 4;
                            float src_value[4];
                            for (int k = 0; k < cn; ++k) {
                                src_value[0] = src[srcIndex + k];
                                src_value[1] = src[srcIndex + cn + k];
                                src_value[2] = src[srcIndex + inWidthStride + k];
                                src_value[3] = src[srcIndex + inWidthStride + cn + k];

                                float sum = 0;
                                //linear interpolation
                                sum += src_value[0] * p_short_tab[0];
                                sum += src_value[1] * p_short_tab[1];
                                sum += src_value[2] * p_short_tab[2];
                                sum += src_value[3] * p_short_tab[3];
                                dst[dstIndex + k] = sum / INTER_REMAP_COEF_SCALE;
                            } 
                        }
                        else
                        {
                            continue;
                        }
                    }
                    else if(borderMode == BORDER_TYPE_REPLICATE)
                    {
                        int sx0 = clip(srcX, 0, inWidth - 1);
                        int sy0 = clip(srcY, 0, inHeight - 1);
                        int sx1 = clip((srcX + 1), 0, inWidth - 1);
                        int sy1 = clip((srcY + 1), 0, inHeight - 1);
                        
                        const float *t0 = src + sy0 * inWidthStride + sx0 * cn;
                        const float *t1 = src + sy0 * inWidthStride + sx1 * cn;
                        const float *t2 = src + sy1 * inWidthStride + sx0 * cn;
                        const float *t3 = src + sy1 * inWidthStride + sx1 * cn;

                        const short *p_short_tab = short_tab + XY_DEC[y1 * bw + x1] * 4;
                        for (int k = 0; k < cn; ++k) {
                            float sum = 0;
                            //linear interpolation
                            sum += t0[k] * p_short_tab[0];
                            sum += t1[k] * p_short_tab[1];
                            sum += t2[k] * p_short_tab[2];
                            sum += t3[k] * p_short_tab[3];
                            dst[dstIndex + k] = sum / INTER_REMAP_COEF_SCALE;
                        }
                    }
                }
            }
        }
    }

    free(short_tab);	
    free(adelta);
    free(bdelta);
}

template<>
::ppl::common::RetCode  WarpAffineNearestPoint<float, 1>(
    int inHeight, int inWidth, int inWidthStride, const float *inData, 
    int outHeight, int outWidth, int outWidthStride, float *outData, 
    const float *affineMatrix, BorderType border_type, float borderValue) {

    return warpAffine_nearest<float>(outData, inData, 
        inHeight, inWidth, inWidthStride, 
        outHeight, outWidth, outWidthStride, 
        affineMatrix, 1, border_type, borderValue);
}

template<>
::ppl::common::RetCode  WarpAffineNearestPoint<float, 3>(
    int inHeight, int inWidth, int inWidthStride, const float *inData, 
    int outHeight, int outWidth, int outWidthStride, float *outData, 
    const float *affineMatrix, BorderType border_type, float borderValue) {

    return warpAffine_nearest<float>(outData, inData, 
        inHeight, inWidth, inWidthStride, 
        outHeight, outWidth, outWidthStride, 
        affineMatrix, 3, border_type, borderValue);
}

template<>
::ppl::common::RetCode  WarpAffineNearestPoint<float, 4>(
    int inHeight, int inWidth, int inWidthStride, const float *inData, 
    int outHeight, int outWidth, int outWidthStride, float *outData, 
    const float *affineMatrix, BorderType border_type, float borderValue) {

    return warpAffine_nearest<float>(outData, inData, 
        inHeight, inWidth, inWidthStride, 
        outHeight, outWidth, outWidthStride, 
        affineMatrix, 4, border_type, borderValue);
}

template<>
::ppl::common::RetCode  WarpAffineLinear<float, 1>(
    int inHeight, int inWidth, int inWidthStride, const float *inData, 
    int outHeight, int outWidth, int outWidthStride, float *outData, 
    const float *affineMatrix, BorderType border_type, float borderValue) {

    return warpAffine_linear_float(outData, inData, 
        inHeight, inWidth, inWidthStride, 
        outHeight, outWidth, outWidthStride, 
        affineMatrix, 1, border_type, borderValue);
}

template<>
::ppl::common::RetCode  WarpAffineLinear<float, 3>(
    int inHeight, int inWidth, int inWidthStride, const float *inData, 
    int outHeight, int outWidth, int outWidthStride, float *outData, 
    const float *affineMatrix, BorderType border_type, float borderValue) {

    return warpAffine_linear_float(outData, inData, 
        inHeight, inWidth, inWidthStride, 
        outHeight, outWidth, outWidthStride, 
        affineMatrix, 3, border_type, borderValue);
}

template<>
::ppl::common::RetCode  WarpAffineLinear<float, 4>(
    int inHeight, int inWidth, int inWidthStride, const float *inData, 
    int outHeight, int outWidth, int outWidthStride, float *outData, 
    const float *affineMatrix, BorderType border_type, float borderValue) {

    return warpAffine_linear_float(outData, inData, 
        inHeight, inWidth, inWidthStride, 
        outHeight, outWidth, outWidthStride, 
        affineMatrix, 4, border_type, borderValue);
}

}
}
} // namespace ppl::cv::aarch64
