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

#include "ppl/cv/aarch64/arithmetic.h"
#include "ppl/cv/aarch64/morph.hpp"

#include <assert.h>
#include <stdio.h>
#include <cmath>
#include <float.h>
#include "string.h"
#include "typetraits.hpp"
#include <arm_neon.h>
#include "ppl/common/log.h"
#include "common.hpp"
#include <iostream>

namespace ppl {
namespace cv {
namespace aarch64 {


#define p(x) LOG(INFO) << " "#x;vprint<nc, v_dt>(x);
           

inline void stq(float *dst, float32x4_t t) {vst1q_f32(dst, t);}
inline void stq(float *dst, float32x4x3_t t) {vst3q_f32(dst, t);}
inline void stq(float *dst, float32x4x4_t t) {vst4q_f32(dst, t);}

template<int nc, typename T>
void vprint(T t){
    float32_t data[4*nc] = {};
    stq(data, t);
    for(int i = 0; i < nc; i++){
        for(int j = 0; j < 4; j++){
            std::cout << data[i*4 + j] << " ";
        }
    }
    std::cout << std::endl;
}

template<int nc, typename T, typename V>
inline void vdupq(T &src, V value) {
    for(int i = 0; i < nc; i++){
        src.val[i] = vdupq_n_f32(value);
    }
}

template <>
inline void vdupq<1, float32x4_t, float>(float32x4_t &src, float value) {
    src = vdupq_n_f32(value);
}


template<int nc, typename T>
inline void vldnq(T& src, const float *ptr);

template <>
inline void vldnq<1, float32x4_t>(float32x4_t& src, const float *ptr){src = vld1q_f32(ptr);}

template <>
inline void vldnq<3, float32x4x3_t>(float32x4x3_t& src, const float *ptr){src = vld3q_f32(ptr);}

template <>
inline void vldnq<4, float32x4x4_t>(float32x4x4_t& src, const float *ptr){src = vld4q_f32(ptr);}

template <class morphOp, int32_t nc, typename T>
inline void compare(T& tnext, T& v_up, T& v_mid, T& v_down)
{
    morphOp vop;
    for(int32_t i = 0; i < nc; i++) {
        tnext.val[i] = vop(vop(v_up.val[i], v_mid.val[i]), v_down.val[i]);
    }
}

template <>
inline void compare<DilateVecOp, 1, float32x4_t>(float32x4_t& tnext, float32x4_t& v_up, float32x4_t& v_mid, float32x4_t& v_down)
{
    DilateVecOp vop;
    tnext = vop(vop(v_up, v_mid), v_down);
}

template <>
inline void compare<ErodeVecOp, 1, float32x4_t>(float32x4_t& tnext, float32x4_t& v_up, float32x4_t& v_mid, float32x4_t& v_down)
{
    ErodeVecOp vop;
    tnext = vop(vop(v_up, v_mid), v_down);
}

template <int32_t nc, typename T>
inline void vextq(T& t1, T& t2, T& t3, int n)
{
    for(int32_t i = 0; i < nc; i++) {
        t1.val[i] = vextq_f32(t2.val[i], t3.val[i], n);
    }
}

template <>
inline void vextq<1, float32x4_t>(float32x4_t& t1, float32x4_t& t2, float32x4_t& t3, int n)
{
    t1 = vextq_f32(t2, t3, n);
}


#define VLEN 16 // 16 bytes = 128 bits for  reg
template <typename T>
inline T *getRowPtr(T *base, int32_t stride, int32_t row)
{
    float *baseRaw = const_cast<float *>(reinterpret_cast<const float *>(base));
    return reinterpret_cast<T *>(baseRaw + row * stride);
}

template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphRow(typename DT<nc, float>::vec_DT *tprev, typename DT<nc, float>::vec_DT &tcurr, typename DT<nc, float>::vec_DT *tnext, const float *srcCenterRow, int32_t srcStride, float *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, float borderValue = 0)
{
    using v_dt = typename DT<nc, float>::vec_DT;
    using cmpptr = void (*)(v_dt&, v_dt&, v_dt&, v_dt&);
    cmpptr vop = compare<morphOp, nc, v_dt>;
    using vldptr = void (*)(v_dt&, const float*);
    vldptr vld = vldnq<nc, v_dt>;
    v_dt v_border;
    vdupq<nc, v_dt, float>(v_border, borderValue);
    constexpr int32_t kernel_radius   = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num  = kernel_radius;
    constexpr int32_t v_elem          = VLEN / sizeof(float);
    constexpr int8_t invalid_byte_len = VLEN % (nc * sizeof(float));
    switch (kernel_len) {
        case 3: {
            v_dt v_up, v_mid, v_down;
            v_dt t_left, t_mid, t_right;
            if(rowIdx == 0){v_up = v_border;}else{vldnq<nc, v_dt>(v_up, srcCenterRow - srcStride);};
            vldnq<nc, v_dt>(v_mid, srcCenterRow);
            if(rowIdxInv == 0){v_down = v_border;}else{vldnq<nc, v_dt>(v_down, srcCenterRow + srcStride);};

            compare<morphOp, nc, v_dt>(tnext[0], v_up, v_mid, v_down);

            vextq<nc, v_dt>(t_left, tprev[0], tcurr, 3); 
            t_mid = tcurr;
            vextq<nc, v_dt>(t_right, tcurr, tnext[0], 1);
            
            compare<morphOp, nc, v_dt>(t_mid, t_left, t_mid, t_right);
            stq(drow, t_mid);
        } break;
        case 5: {
            v_dt v_up0, v_up1, v_mid, v_down0, v_down1;
            if(rowIdx < 2){v_up0 = v_border;}else{vldnq<nc, v_dt>(v_up0, srcCenterRow - 2 * srcStride + v_elem * nc * (radius_vec_num - 1));};
            if(rowIdx < 1){v_up1 = v_border;}else{vldnq<nc, v_dt>(v_up1, srcCenterRow - 1 * srcStride + v_elem * nc * (radius_vec_num - 1));};
            vldnq<nc, v_dt>(v_mid, srcCenterRow + v_elem * nc * (radius_vec_num - 1));
            if(rowIdxInv < 1){v_down0 = v_border;}else{vldnq<nc, v_dt>(v_down0, srcCenterRow + 1 * srcStride + v_elem * nc * (radius_vec_num - 1));};
            if(rowIdxInv < 2){v_down1 = v_border;}else{vldnq<nc, v_dt>(v_down1, srcCenterRow + 2 * srcStride + v_elem * nc * (radius_vec_num - 1));};
            vop(tnext[radius_vec_num - 1], v_up0, v_up1, v_mid);
            vop(tnext[radius_vec_num - 1], tnext[radius_vec_num - 1], v_down0, v_down1);

            v_dt t_left0, t_left1, t_mid, t_right0, t_right1;
            vextq<nc, v_dt>(t_left0, tprev[1], tcurr, 2); 
            vextq<nc, v_dt>(t_left1, tprev[1], tcurr, 3); 
            t_mid = tcurr;
            vextq<nc, v_dt>(t_right0, tcurr, tnext[0], 1);
            vextq<nc, v_dt>(t_right1, tcurr, tnext[0], 2);
            
            vop(t_mid, t_mid, t_left0, t_left1);
            vop(t_mid, t_mid, t_right0, t_right1);
            stq(drow, t_mid);
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphRowLast(typename DT<nc, float>::vec_DT *tprev, typename DT<nc, float>::vec_DT &tcurr, typename DT<nc, float>::vec_DT *tnext, const float *srcCenterRow, int32_t srcStride, float *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, float borderValue = 0)
{
    using v_dt = typename DT<nc, float>::vec_DT;
    using cmpptr = void (*)(v_dt&, v_dt&, v_dt&, v_dt&);
    cmpptr vop = compare<morphOp, nc, v_dt>;
    using vldptr = void (*)(v_dt&, const float*);
    vldptr vld = vldnq<nc, v_dt>;
    v_dt v_border;
    vdupq<nc, v_dt, float>(v_border, borderValue);
    constexpr int32_t kernel_radius   = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num  = kernel_radius;
    constexpr int32_t v_elem          = VLEN / sizeof(float) / nc;
    constexpr int8_t invalid_byte_len = VLEN % (nc * sizeof(float));
    int32_t bias = colIdxInv + 1;
    switch (kernel_len) {
        case 3: {
            v_dt v_up, v_mid, v_down;
            v_dt t_left, t_mid, t_right;
            tnext[0] = v_border;
            v_dt F_min;
            vdupq<nc, v_dt, float>(F_min, FLT_MIN);
            if(bias == 3){
                vextq<nc, v_dt>(t_left, tprev[0], tcurr, 2);
                vextq<nc, v_dt>(t_mid, tprev[0], tcurr, 3);
                vextq<nc, v_dt>(t_right, t_mid, v_border, 1);
            } else if(bias == 2) {
                vextq<nc, v_dt>(t_left, tprev[0], tcurr, 1);
                vextq<nc, v_dt>(t_mid, tprev[0], tcurr, 2);
                vextq<nc, v_dt>(t_right, t_mid, v_border, 1);
            } else if (bias == 1) {
                vextq<nc, v_dt>(t_left, tprev[0], tcurr, 0);
                vextq<nc, v_dt>(t_mid, tprev[0], tcurr, 1);
                vextq<nc, v_dt>(t_right, t_mid, v_border, 1);
            } else {
                if(rowIdx == 0){v_up = v_border;}else{vldnq<nc, v_dt>(v_up, srcCenterRow - nc * 2 * 4 - nc - srcStride);};
                vldnq<nc, v_dt>(v_mid, srcCenterRow - nc * 2 * 4 - nc);
                if(rowIdxInv == 0){v_down = v_border;}else{vldnq<nc, v_dt>(v_down, srcCenterRow - nc * 2 * 4 - nc + srcStride);};
                compare<morphOp, nc, v_dt>(t_left, v_up, v_mid, v_down);
                vextq<nc, v_dt>(t_mid, tprev[0], tcurr, 0);
                vextq<nc, v_dt>(t_right, t_mid, v_border, 1);
            }
            compare<morphOp, nc, v_dt>(t_mid, t_left, t_mid, t_right);
            stq(drow - (4 - bias) * nc, t_mid);
        } break;
        case 5: {
            v_dt v_up0, v_up1, v_mid, v_down0, v_down1;
            v_dt t_left0, t_left1, t_mid, t_right0, t_right1;

            tnext[0] = v_border;
            v_dt F_min;
            vdupq<nc, v_dt, float>(F_min, FLT_MIN);
            if(bias == 3){
                vextq<nc, v_dt>(t_left0, tprev[1], tcurr, 1);
                vextq<nc, v_dt>(t_left1, tprev[1], tcurr, 2);
                vextq<nc, v_dt>(t_mid, tprev[1], tcurr, 3);
                vextq<nc, v_dt>(t_right0, t_mid, v_border, 1);
                vextq<nc, v_dt>(t_right1, t_mid, v_border, 2);

            } else if(bias == 2) {
                vextq<nc, v_dt>(t_left0, tprev[1], tcurr, 0);
                vextq<nc, v_dt>(t_left1, tprev[1], tcurr, 1);
                vextq<nc, v_dt>(t_mid, tprev[1], tcurr, 2);
                vextq<nc, v_dt>(t_right0, t_mid, v_border, 1);
                vextq<nc, v_dt>(t_right1, t_mid, v_border, 2);
            } else if (bias == 1) {
                vextq<nc, v_dt>(t_left0, tprev[0], tprev[1], 3);
                vextq<nc, v_dt>(t_left1, tprev[1], tcurr, 0);
                vextq<nc, v_dt>(t_mid, tprev[1], tcurr, 1);
                vextq<nc, v_dt>(t_right0, t_mid, v_border, 1);
                vextq<nc, v_dt>(t_right1, t_mid, v_border, 2);
            } else {
                // p(tprev[0])
                // p(tprev[1])
                // p(tcurr)
                vextq<nc, v_dt>(t_left0, tprev[0], tprev[1], 2);
                vextq<nc, v_dt>(t_left1, tprev[0], tprev[1], 3);
                t_mid = tprev[1];
                // vextq<nc, v_dt>(t_mid, tprev[1], tcurr, 3);
                vextq<nc, v_dt>(t_right0, t_mid, v_border, 1);
                vextq<nc, v_dt>(t_right1, t_mid, v_border, 2);
                // p(t_left0)
                // p(t_left1)
                // p(t_mid)
                // p(t_right0)
                // p(t_right1)
            }
            vop(t_mid, t_mid, t_left0, t_left1);
            vop(t_mid, t_mid, t_right0, t_right1);

            stq(drow - (4 - bias) * nc, t_mid);
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphFirstCol(typename DT<nc, float>::vec_DT &tcurr, typename DT<nc, float>::vec_DT *tnext, const float *srcCenterRow, int32_t srcStride, int32_t rowIdx, int32_t rowIdxInv, float borderValue = 0)
{
    using v_dt = typename DT<nc, float>::vec_DT;
    using cmpptr = void (*)(v_dt&, v_dt&, v_dt&, v_dt&);
    cmpptr vop = compare<morphOp, nc, v_dt>;
    using vldptr = void (*)(v_dt&, const float*);
    vldptr vld = vldnq<nc, v_dt>;
    v_dt v_border;
    vdupq<nc, v_dt, float>(v_border, borderValue);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = kernel_radius;
    constexpr int32_t v_elem         = VLEN / sizeof(float);

    switch (kernel_len) {
        case 3: {
            v_dt v_up, v_mid, v_down;
            if(rowIdx == 0){v_up = v_border;}else{vld(v_up, srcCenterRow - srcStride);};
            vld(v_mid, srcCenterRow);
            if(rowIdxInv == 0){v_down = v_border;}else{vld(v_down, srcCenterRow + srcStride);};
            vop(tnext[0], v_up, v_mid, v_down);
            tcurr    = v_border;
        } break;
        case 5: {
            for (int32_t i = 0; i < radius_vec_num; i++) {
                v_dt v_up0, v_up1, v_mid, v_down0, v_down1;
                
                if(rowIdx < 2){v_up0 = v_border;}else{vld(v_up0, srcCenterRow - 2 * srcStride + v_elem * nc * i);};
                if(rowIdx < 1){v_up1 = v_border;}else{vld(v_up1, srcCenterRow - 1 * srcStride + v_elem * nc * i);};
                vld(v_mid, srcCenterRow + v_elem * nc * i);
                if(rowIdxInv < 1){v_down0 = v_border;}else{vld(v_down0, srcCenterRow + 1 * srcStride + v_elem * nc * i);};
                if(rowIdxInv < 2){v_down1 = v_border;}else{vld(v_down1, srcCenterRow + 2 * srcStride + v_elem * nc * i);};
                vop(tnext[i], v_up0, v_up1, v_mid);
                vop(tnext[i], tnext[i], v_down0, v_down1);
            }
            tcurr = v_border;
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
void morph_f32(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue)
{
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t v_elem         = VLEN / sizeof(float);
    constexpr int32_t radius_vec_num = kernel_radius;
    using v_dt = typename DT<nc, float>::vec_DT;
    v_dt tcurr, tprev[radius_vec_num], tnext[radius_vec_num], v_border;
    vdupq<nc, v_dt, float>(v_border, borderValue);

    prefetch(srcBase);
    for(int y = 0; y < height; y++){
        const float *srow = getRowPtr(srcBase, srcStride, y);
        float *drow       = getRowPtr(dstBase, dstStride, y);
        prefetch(srow);
        prefetch(srow + srcStride);
        prefetch(drow);
        MorphFirstCol<morphOp, nc, kernel_len>(tcurr, tnext, srow, srcStride, y, height - 1 - y, borderValue);
        int32_t x = v_elem;
        for (; x <= width ; x += v_elem) {
            // shift
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tprev[i - 1] = tprev[i];
            }
            tprev[radius_vec_num - 1] = tcurr;
            tcurr                     = tnext[0];
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tnext[i - 1] = tnext[i];
            }
            MorphRow<morphOp, nc, kernel_len>(tprev, tcurr, tnext, srow + x * nc, srcStride, drow, y, height - 1 - y, x - v_elem, width - 1 - (x - v_elem), borderValue);
            
            drow += v_elem * nc;
        }
        if (x - v_elem <= width) {
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tprev[i - 1] = tprev[i];
            }
            tprev[radius_vec_num - 1] = tcurr;
            tcurr                     = tnext[0];
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tnext[i - 1] = tnext[i];
            }
            MorphRowLast<morphOp, nc, kernel_len>(tprev, tcurr, tnext, srow + x * nc, srcStride, drow, y, height - 1 - y, x - v_elem, width - 1 - (x - v_elem), borderValue);
        }
    }
}

template void morph_f32<DilateVecOp, 1, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 1, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 3, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 3, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 4, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 4, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);

template void morph_f32<ErodeVecOp, 1, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 1, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 3, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 3, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 4, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 4, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);

}
}
} // namespace ppl::cv::aarch64

