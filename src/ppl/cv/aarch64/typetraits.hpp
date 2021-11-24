#ifndef __ST_HPC_PPL_CV_ARM_TYPETRAITS_H_
#define __ST_HPC_PPL_CV_ARM_TYPETRAITS_H_
#include <stdint.h>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace aarch64 {

typedef unsigned char uchar;

template <int nc, typename T>
struct DT;

template <>
struct DT<1, unsigned char>
{
    typedef uint8x16_t vec_DT;
};
template <>
struct DT<2, unsigned char>
{
    typedef uint8x16x2_t vec_DT;
};
template <>
struct DT<3, unsigned char>
{
    typedef uint8x16x3_t vec_DT;
};
template <>
struct DT<4, unsigned char>
{
    typedef uint8x16x4_t vec_DT;
};
template <>
struct DT<1, float>
{
    typedef float32x4_t vec_DT;
};
template <>
struct DT<2, float>
{
    typedef float32x4x2_t vec_DT;
};
template <>
struct DT<3, float>
{
    typedef float32x4x3_t vec_DT;
};
template <>
struct DT<4, float>
{
    typedef float32x4x4_t vec_DT;
};

template <>
struct DT<1, int>
{
    typedef int32x4_t vec_DT;
};
template <>
struct DT<2, int>
{
    typedef int32x4x2_t vec_DT;
};
template <>
struct DT<3, int>
{
    typedef int32x4x3_t vec_DT;
};
template <>
struct DT<4, int>
{
    typedef int32x4x4_t vec_DT;
};

template <>
struct DT<1, short>
{
    typedef int16x8_t vec_DT;
};
template <>
struct DT<2, short>
{
    typedef int16x8x2_t vec_DT;
};
template <>
struct DT<3, short>
{
    typedef int16x8x3_t vec_DT;
};
template <>
struct DT<4, short>
{
    typedef int16x8x4_t vec_DT;
};

template <int nc, typename T>
struct DT16;
template <>
struct DT16<1, unsigned char>
{
    typedef uint8x16_t vec_DT;
};
template <>
struct DT16<2, unsigned char>
{
    typedef uint8x16x2_t vec_DT;
};
template <>
struct DT16<3, unsigned char>
{
    typedef uint8x16x3_t vec_DT;
};
template <>
struct DT16<4, unsigned char>
{
    typedef uint8x16x4_t vec_DT;
};

template <int nc, typename Tptr, typename T>
inline void vstx_u8_f32(Tptr *ptr, T vec);
template <int nc, typename Tptr, typename T>
inline void vst_s16(Tptr *ptr, T vec);
template <int nc, typename Tptr, typename T>
inline T vldx_u8_f32(const Tptr *ptr);

template <int nc, typename T>
inline void vzero_u8_f32(T *ptr);
template <int nc, typename T>
inline void vzero_s16(T *ptr);
template <int nc, typename T, typename T_val>
inline void vmla_u8_f32(T *dst, T src0, T_val src1);
template <int nc, typename T, typename T_val>
inline void vmlaq_int16(T *dst, T src0, T_val src1);
template <int nc, typename T>
inline void vadq_s16(T *dst, T src0, T src1);
template <int nc, typename T>
inline void vsuq_s16(T *dst, T src0, T src1);
template <int nc, typename T>
inline void vadd_u8_f32(T *dst, T src);
template <int nc, typename T>
inline void vzero_i32(T *ptr);

template <int nc, typename T>
inline void mla_ptr_pixel(T *dst, const T *src0, T src1);
template <int nc, typename T>
inline void stx_pixel(T *ptr, T *val);
template <int nc, typename TSrc, typename TDst>
inline void changeDT_u8x8_f32x4(const TSrc src, TDst &dstHigh, TDst &dstLow);
template <int nc, typename TSrc, typename TDst>
inline void changeDT_u8x8_s16x8(const TSrc src, TDst &dst);
template <int nc, typename TSrc, typename TDst>
inline void changeDT_f32x4_u8x8(const TSrc srcHigh, const TSrc srcLow, TDst &dst);
template <int nc, typename TSrc, typename TDst>
inline void changeDT_f32x4_s16x8(const TSrc srcHigh, const TSrc srcLow, TDst &dst);


template <int nc, typename T1, typename T2, int idx>
inline void get_pixel(T1 *dst, T2 src);

template <>
inline void mla_ptr_pixel<1, float>(float *dst, const float *src0, float src1)
{
    *dst += *src0 * src1;
}
template <>
inline void mla_ptr_pixel<3, float>(float *dst, const float *src0, float src1)
{
    *dst += *src0 * src1;
    *(dst + 1) += *(src0 + 1) * src1;
    *(dst + 2) += *(src0 + 2) * src1;
}
template <>
inline void mla_ptr_pixel<4, float>(float *dst, const float *src0, float src1)
{
    *dst += *src0 * src1;
    *(dst + 1) += *(src0 + 1) * src1;
    *(dst + 2) += *(src0 + 2) * src1;
    *(dst + 3) += *(src0 + 3) * src1;
}

template <>
inline void mla_ptr_pixel<1, int16_t>(int16_t *dst, const int16_t *src0, int16_t src1)
{
    *dst += *src0 * src1;
}
template <>
inline void mla_ptr_pixel<3, int16_t>(int16_t *dst, const int16_t *src0, int16_t src1)
{
    *dst += *src0 * src1;
    *(dst + 1) += *(src0 + 1) * src1;
    *(dst + 2) += *(src0 + 2) * src1;
}
template <>
inline void mla_ptr_pixel<4, int16_t>(int16_t *dst, const int16_t *src0, int16_t src1)
{
    *dst += *src0 * src1;
    *(dst + 1) += *(src0 + 1) * src1;
    *(dst + 2) += *(src0 + 2) * src1;
    *(dst + 3) += *(src0 + 3) * src1;
}

template <>
inline void stx_pixel<1, float>(float *ptr, float *val)
{
    *ptr = *val;
}
template <>
inline void stx_pixel<3, float>(float *ptr, float *val)
{
    *ptr = *val;
    *(ptr + 1) = *(val + 1);
    *(ptr + 2) = *(val + 2);
}
template <>
inline void stx_pixel<4, float>(float *ptr, float *val)
{
    *ptr = *val;
    *(ptr + 1) = *(val + 1);
    *(ptr + 2) = *(val + 2);
    *(ptr + 3) = *(val + 3);
}
template <>
inline void stx_pixel<1, uchar>(uchar *ptr, uchar *val)
{
    *ptr = *val;
}
template <>
inline void stx_pixel<3, uchar>(uchar *ptr, uchar *val)
{
    *ptr = *val;
    *(ptr + 1) = *(val + 1);
    *(ptr + 2) = *(val + 2);
}
template <>
inline void stx_pixel<4, uchar>(uchar *ptr, uchar *val)
{
    *ptr = *val;
    *(ptr + 1) = *(val + 1);
    *(ptr + 2) = *(val + 2);
    *(ptr + 3) = *(val + 3);
}

template <>
inline void stx_pixel<1, int16_t>(int16_t *ptr, int16_t *val)
{
    *ptr = *val;
}
template <>
inline void stx_pixel<3, int16_t>(int16_t *ptr, int16_t *val)
{
    *ptr = *val;
    *(ptr + 1) = *(val + 1);
    *(ptr + 2) = *(val + 2);
}
template <>
inline void stx_pixel<4, int16_t>(int16_t *ptr, int16_t *val)
{
    *ptr = *val;
    *(ptr + 1) = *(val + 1);
    *(ptr + 2) = *(val + 2);
    *(ptr + 3) = *(val + 3);
}

template <int nc, typename Tptr>
inline void add_pixel(Tptr *sum, const Tptr *img);
template <>
inline void add_pixel<1, float>(float *sum, const float *img)
{
    float tmp_val = *img;
    *sum += tmp_val;
}
template <>
inline void add_pixel<3, float>(float *sum, const float *img)
{
    float tmp_val0 = *img;
    float tmp_val1 = *(img + 1);
    float tmp_val2 = *(img + 2);
    *sum += tmp_val0;
    *(sum + 1) += tmp_val1;
    *(sum + 2) += tmp_val2;
}
template <>
inline void add_pixel<4, float>(float *sum, const float *img)
{
    float tmp_val0 = *img;
    float tmp_val1 = *(img + 1);
    float tmp_val2 = *(img + 2);
    float tmp_val3 = *(img + 3);
    *sum += tmp_val0;
    *(sum + 1) += tmp_val1;
    *(sum + 2) += tmp_val2;
    *(sum + 3) += tmp_val3;
}

template <>
inline void add_pixel<1, int32_t>(int32_t *sum, const int32_t *img)
{
    *sum += *img;
}
template <>
inline void add_pixel<3, int32_t>(int32_t *sum, const int32_t *img)
{
    *sum += *img;
    *(sum + 1) += *(img + 1);
    *(sum + 2) += *(img + 2);
}
template <>
inline void add_pixel<4, int32_t>(int32_t *sum, const int32_t *img)
{
    *sum += *img;
    *(sum + 1) += *(img + 1);
    *(sum + 2) += *(img + 2);
    *(sum + 3) += *(img + 3);
}

template <int nc, typename T1, typename T2>
inline void addw_pixel(T1 *sum, const T2 *img);
template <>
inline void addw_pixel<1, int32_t, uchar>(int32_t *sum, const uchar *img)
{
    *sum += *img;
}
template <>
inline void addw_pixel<3, int32_t, uchar>(int32_t *sum, const uchar *img)
{
    *sum += *img;
    *(sum + 1) += *(img + 1);
    *(sum + 2) += *(img + 2);
}
template <>
inline void addw_pixel<4, int32_t, uchar>(int32_t *sum, const uchar *img)
{
    *sum += *img;
    *(sum + 1) += *(img + 1);
    *(sum + 2) += *(img + 2);
    *(sum + 3) += *(img + 3);
}

template <int nc, typename Tptr>
inline void add_pixel(Tptr *sum, const Tptr *src0, const Tptr *src1);
template <>
inline void add_pixel<1, float>(float *sum, const float *src0, const float *src1)
{
    *(sum) = *(src0) + *(src1);
}
template <>
inline void add_pixel<3, float>(float *sum, const float *src0, const float *src1)
{
    *(sum) = *(src0) + *(src1);
    *(sum + 1) = *(src0 + 1) + *(src1 + 1);
    *(sum + 2) = *(src0 + 2) + *(src1 + 2);
}
template <>
inline void add_pixel<4, float>(float *sum, const float *src0, const float *src1)
{
    *(sum) = *(src0) + *(src1);
    *(sum + 1) = *(src0 + 1) + *(src1 + 1);
    *(sum + 2) = *(src0 + 2) + *(src1 + 2);
    *(sum + 3) = *(src0 + 3) + *(src1 + 3);
}

template <>
inline void add_pixel<1, int>(int *sum, const int *src0, const int *src1)
{
    *(sum) = *(src0) + *(src1);
}
template <>
inline void add_pixel<3, int>(int *sum, const int *src0, const int *src1)
{
    *(sum) = *(src0) + *(src1);
    *(sum + 1) = *(src0 + 1) + *(src1 + 1);
    *(sum + 2) = *(src0 + 2) + *(src1 + 2);
}
template <>
inline void add_pixel<4, int>(int *sum, const int *src0, const int *src1)
{
    *(sum) = *(src0) + *(src1);
    *(sum + 1) = *(src0 + 1) + *(src1 + 1);
    *(sum + 2) = *(src0 + 2) + *(src1 + 2);
    *(sum + 3) = *(src0 + 3) + *(src1 + 3);
}

template <int nc, typename T>
inline void mul_pixel(T *sum, const T val);
template <>
inline void mul_pixel<1, float>(float *sum, const float val)
{
    *sum *= val;
}
template <>
inline void mul_pixel<3, float>(float *sum, const float val)
{
    *sum *= val;
    *(sum + 1) *= val;
    *(sum + 2) *= val;
}
template <>
inline void mul_pixel<4, float>(float *sum, const float val)
{
    *sum *= val;
    *(sum + 1) *= val;
    *(sum + 2) *= val;
    *(sum + 3) *= val;
}

template <int nc, typename Tptr>
inline void ldx_pixel(Tptr *val, const Tptr *img);
template <>
inline void ldx_pixel<1, float>(float *val, const float *img)
{
    *(val) = *(img);
}
template <>
inline void ldx_pixel<3, float>(float *val, const float *img)
{
    *(val) = *(img);
    *(val + 1) = *(img + 1);
    *(val + 2) = *(img + 2);
}
template <>
inline void ldx_pixel<4, float>(float *val, const float *img)
{
    *(val) = *(img);
    *(val + 1) = *(img + 1);
    *(val + 2) = *(img + 2);
    *(val + 3) = *(img + 3);
}

template <int nc, typename Tptr>
inline void out_pixel(const Tptr *sum, Tptr *img);
template <>
inline void out_pixel<1, float>(const float *sum, float *img)
{
    *(img) = *(sum);
}
template <>
inline void out_pixel<3, float>(const float *sum, float *img)
{
    *(img) = *(sum);
    *(img + 1) = *(sum + 1);
    *(img + 2) = *(sum + 2);
}
template <>
inline void out_pixel<4, float>(const float *sum, float *img)
{
    *(img) = *(sum);
    *(img + 1) = *(sum + 1);
    *(img + 2) = *(sum + 2);
    *(img + 3) = *(sum + 3);
}

template <int nc, typename Tptr>
inline void out_pixel(const Tptr *sum, Tptr *img);
template <>
inline void out_pixel<1, int32_t>(const int32_t *sum, int32_t *img)
{
    *(img) = *(sum);
}
template <>
inline void out_pixel<3, int32_t>(const int32_t *sum, int32_t *img)
{
    *(img) = *(sum);
    *(img + 1) = *(sum + 1);
    *(img + 2) = *(sum + 2);
}
template <>
inline void out_pixel<4, int32_t>(const int32_t *sum, int32_t *img)
{
    *(img) = *(sum);
    *(img + 1) = *(sum + 1);
    *(img + 2) = *(sum + 2);
    *(img + 3) = *(sum + 3);
}

//Neon operations
template <>
inline void vstx_u8_f32<1, unsigned char, uint8x8_t>(unsigned char *ptr, uint8x8_t vec) { vst1_u8(ptr, vec); }
template <>
inline void vstx_u8_f32<2, uchar, uint8x8x2_t>(uchar *ptr, uint8x8x2_t vec) { vst2_u8(ptr, vec); }
template <>
inline void vstx_u8_f32<3, uchar, uint8x8x3_t>(uchar *ptr, uint8x8x3_t vec) { vst3_u8(ptr, vec); }
template <>
inline void vstx_u8_f32<4, uchar, uint8x8x4_t>(uchar *ptr, uint8x8x4_t vec) { vst4_u8(ptr, vec); }
template <>
inline void vstx_u8_f32<1, float, float32x4_t>(float *ptr, float32x4_t vec) { vst1q_f32(ptr, vec); }
template <>
inline void vstx_u8_f32<2, float, float32x4x2_t>(float *ptr, float32x4x2_t vec) { vst2q_f32(ptr, vec); }
template <>
inline void vstx_u8_f32<3, float, float32x4x3_t>(float *ptr, float32x4x3_t vec) { vst3q_f32(ptr, vec); }
template <>
inline void vstx_u8_f32<4, float, float32x4x4_t>(float *ptr, float32x4x4_t vec) { vst4q_f32(ptr, vec); }
template <>
inline void vstx_u8_f32<3, uchar, uint8x16x3_t>(uchar *ptr, uint8x16x3_t vec) { vst3q_u8(ptr, vec); }
template <>
inline void vstx_u8_f32<4, uchar, uint8x16x4_t>(uchar *ptr, uint8x16x4_t vec) { vst4q_u8(ptr, vec); }
template <>
inline void vst_s16<1, int16_t, int16x8_t>(int16_t *ptr, int16x8_t vec) { vst1q_s16(ptr, vec); }
template <>
inline void vst_s16<3, int16_t, int16x8x3_t>(int16_t *ptr, int16x8x3_t vec) { vst3q_s16(ptr, vec); }
template <>
inline void vst_s16<4, int16_t, int16x8x4_t>(int16_t *ptr, int16x8x4_t vec) { vst4q_s16(ptr, vec); }

template <>
inline uint8x8_t vldx_u8_f32<1, unsigned char, uint8x8_t>(const unsigned char *ptr) { return vld1_u8(ptr); };
template <>
inline uint8x8x2_t vldx_u8_f32<2, uchar, uint8x8x2_t>(const uchar *ptr) { return vld2_u8(ptr); };
template <>
inline uint8x8x3_t vldx_u8_f32<3, uchar, uint8x8x3_t>(const uchar *ptr) { return vld3_u8(ptr); };
template <>
inline uint8x8x4_t vldx_u8_f32<4, uchar, uint8x8x4_t>(const uchar *ptr) { return vld4_u8(ptr); };
template <>
inline float32x4_t vldx_u8_f32<1, float, float32x4_t>(const float *ptr) { return vld1q_f32(ptr); };
template <>
inline float32x4x2_t vldx_u8_f32<2, float, float32x4x2_t>(const float *ptr) { return vld2q_f32(ptr); };
template <>
inline float32x4x3_t vldx_u8_f32<3, float, float32x4x3_t>(const float *ptr) { return vld3q_f32(ptr); };
template <>
inline float32x4x4_t vldx_u8_f32<4, float, float32x4x4_t>(const float *ptr) { return vld4q_f32(ptr); };

template <>
inline void vzero_u8_f32<1, float32x4_t>(float32x4_t *vec)
{
    *vec = vdupq_n_f32(0.0f);
}
template <>
inline void vzero_u8_f32<3, float32x4x3_t>(float32x4x3_t *vec)
{
    vec->val[0] = vdupq_n_f32(0.0f);
    vec->val[1] = vdupq_n_f32(0.0f);
    vec->val[2] = vdupq_n_f32(0.0f);
}
template <>
inline void vzero_u8_f32<4, float32x4x4_t>(float32x4x4_t *vec)
{
    vec->val[0] = vdupq_n_f32(0.0f);
    vec->val[1] = vdupq_n_f32(0.0f);
    vec->val[2] = vdupq_n_f32(0.0f);
    vec->val[3] = vdupq_n_f32(0.0f);
}

template <>
inline void vzero_i32<1, int32x4_t>(int32x4_t *vec)
{
    *vec = vdupq_n_s32(0.0f);
}
template <>
inline void vzero_i32<3, int32x4x3_t>(int32x4x3_t *vec)
{
    vec->val[0] = vdupq_n_s32(0.0f);
    vec->val[1] = vdupq_n_s32(0.0f);
    vec->val[2] = vdupq_n_s32(0.0f);
}
template <>
inline void vzero_i32<4, int32x4x4_t>(int32x4x4_t *vec)
{
    vec->val[0] = vdupq_n_s32(0.0f);
    vec->val[1] = vdupq_n_s32(0.0f);
    vec->val[2] = vdupq_n_s32(0.0f);
    vec->val[3] = vdupq_n_s32(0.0f);
}

template <>
inline void vzero_s16<1, int16x8_t>(int16x8_t *vec)
{
    *vec = vdupq_n_s16(0);
}
template <>
inline void vzero_s16<3, int16x8x3_t>(int16x8x3_t *vec)
{
    vec->val[0] = vdupq_n_s16(0);
    vec->val[1] = vdupq_n_s16(0);
    vec->val[2] = vdupq_n_s16(0);
}
template <>
inline void vzero_s16<4, int16x8x4_t>(int16x8x4_t *vec)
{
    vec->val[0] = vdupq_n_s16(0);
    vec->val[1] = vdupq_n_s16(0);
    vec->val[2] = vdupq_n_s16(0);
    vec->val[3] = vdupq_n_s16(0);
}

//0
template <>
inline void get_pixel<1, int, int32x4_t, 0>(int *dst, int32x4_t src)
{
    dst[0] = vgetq_lane_s32(src, 0);
}
template <>
inline void get_pixel<3, int, int32x4x3_t, 0>(int *dst, int32x4x3_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 0);
    dst[1] = vgetq_lane_s32(src.val[1], 0);
    dst[2] = vgetq_lane_s32(src.val[2], 0);
}
template <>
inline void get_pixel<4, int, int32x4x4_t, 0>(int *dst, int32x4x4_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 0);
    dst[1] = vgetq_lane_s32(src.val[1], 0);
    dst[2] = vgetq_lane_s32(src.val[2], 0);
    dst[3] = vgetq_lane_s32(src.val[3], 0);
}

//1
template <>
inline void get_pixel<1, int, int32x4_t, 1>(int *dst, int32x4_t src)
{
    dst[0] = vgetq_lane_s32(src, 1);
}
template <>
inline void get_pixel<3, int, int32x4x3_t, 1>(int *dst, int32x4x3_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 1);
    dst[1] = vgetq_lane_s32(src.val[1], 1);
    dst[2] = vgetq_lane_s32(src.val[2], 1);
}
template <>
inline void get_pixel<4, int, int32x4x4_t, 1>(int *dst, int32x4x4_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 1);
    dst[1] = vgetq_lane_s32(src.val[1], 1);
    dst[2] = vgetq_lane_s32(src.val[2], 1);
    dst[3] = vgetq_lane_s32(src.val[3], 1);
}

//2
template <>
inline void get_pixel<1, int, int32x4_t, 2>(int *dst, int32x4_t src)
{
    dst[0] = vgetq_lane_s32(src, 2);
}
template <>
inline void get_pixel<3, int, int32x4x3_t, 2>(int *dst, int32x4x3_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 2);
    dst[1] = vgetq_lane_s32(src.val[1], 2);
    dst[2] = vgetq_lane_s32(src.val[2], 2);
}
template <>
inline void get_pixel<4, int, int32x4x4_t, 2>(int *dst, int32x4x4_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 2);
    dst[1] = vgetq_lane_s32(src.val[1], 2);
    dst[2] = vgetq_lane_s32(src.val[2], 2);
    dst[3] = vgetq_lane_s32(src.val[3], 2);
}

//3
template <>
inline void get_pixel<1, int, int32x4_t, 3>(int *dst, int32x4_t src)
{
    dst[0] = vgetq_lane_s32(src, 3);
}
template <>
inline void get_pixel<3, int, int32x4x3_t, 3>(int *dst, int32x4x3_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 3);
    dst[1] = vgetq_lane_s32(src.val[1], 3);
    dst[2] = vgetq_lane_s32(src.val[2], 3);
}
template <>
inline void get_pixel<4, int, int32x4x4_t, 3>(int *dst, int32x4x4_t src)
{
    dst[0] = vgetq_lane_s32(src.val[0], 3);
    dst[1] = vgetq_lane_s32(src.val[1], 3);
    dst[2] = vgetq_lane_s32(src.val[2], 3);
    dst[3] = vgetq_lane_s32(src.val[3], 3);
}

template <>
inline void vmla_u8_f32<1, float32x4_t, float32x4_t>(float32x4_t *dst, float32x4_t src0, float32x4_t src1)
{
    *dst = vmlaq_f32(*dst, src0, src1);
}
template <>
inline void vmla_u8_f32<3, float32x4x3_t, float32x4_t>(float32x4x3_t *dst, float32x4x3_t src0, float32x4_t src1)
{
    dst->val[0] = vmlaq_f32(dst->val[0], src0.val[0], src1);
    dst->val[1] = vmlaq_f32(dst->val[1], src0.val[1], src1);
    dst->val[2] = vmlaq_f32(dst->val[2], src0.val[2], src1);
}
template <>
inline void vmla_u8_f32<4, float32x4x4_t, float32x4_t>(float32x4x4_t *dst, float32x4x4_t src0, float32x4_t src1)
{
    dst->val[0] = vmlaq_f32(dst->val[0], src0.val[0], src1);
    dst->val[1] = vmlaq_f32(dst->val[1], src0.val[1], src1);
    dst->val[2] = vmlaq_f32(dst->val[2], src0.val[2], src1);
    dst->val[3] = vmlaq_f32(dst->val[3], src0.val[3], src1);
}

template <>
inline void vmlaq_int16<1, int16x8_t, int16x8_t>(int16x8_t *dst, int16x8_t src0, int16x8_t src1)
{
    *dst = vmlaq_s16(*dst, src0, src1);
}
template <>
inline void vmlaq_int16<3, int16x8x3_t, int16x8_t>(int16x8x3_t *dst, int16x8x3_t src0, int16x8_t src1)
{
    dst->val[0] = vmlaq_s16(dst->val[0], src0.val[0], src1);
    dst->val[1] = vmlaq_s16(dst->val[1], src0.val[1], src1);
    dst->val[2] = vmlaq_s16(dst->val[2], src0.val[2], src1);
}
template <>
inline void vmlaq_int16<4, int16x8x4_t, int16x8_t>(int16x8x4_t *dst, int16x8x4_t src0, int16x8_t src1)
{
    dst->val[0] = vmlaq_s16(dst->val[0], src0.val[0], src1);
    dst->val[1] = vmlaq_s16(dst->val[1], src0.val[1], src1);
    dst->val[2] = vmlaq_s16(dst->val[2], src0.val[2], src1);
    dst->val[3] = vmlaq_s16(dst->val[3], src0.val[3], src1);
}

template <>
inline void vadq_s16<1, int16x8_t>(int16x8_t *dst, int16x8_t src0, int16x8_t src1)
{
    *dst = vaddq_s16(src0, src1);
}
template <>
inline void vadq_s16<3, int16x8x3_t>(int16x8x3_t *dst, int16x8x3_t src0, int16x8x3_t src1)
{
    dst->val[0] = vaddq_s16(src0.val[0], src1.val[0]);
    dst->val[1] = vaddq_s16(src0.val[1], src1.val[1]);
    dst->val[2] = vaddq_s16(src0.val[2], src1.val[2]);
}
template <>
inline void vadq_s16<4, int16x8x4_t>(int16x8x4_t *dst, int16x8x4_t src0, int16x8x4_t src1)
{
    dst->val[0] = vaddq_s16(src0.val[0], src1.val[0]);
    dst->val[1] = vaddq_s16(src0.val[1], src1.val[1]);
    dst->val[2] = vaddq_s16(src0.val[2], src1.val[2]);
    dst->val[3] = vaddq_s16(src0.val[3], src1.val[3]);
}

template <>
inline void vsuq_s16<1, int16x8_t>(int16x8_t *dst, int16x8_t src0, int16x8_t src1)
{
    *dst = vsubq_s16(src0, src1);
}
template <>
inline void vsuq_s16<3, int16x8x3_t>(int16x8x3_t *dst, int16x8x3_t src0, int16x8x3_t src1)
{
    dst->val[0] = vsubq_s16(src0.val[0], src1.val[0]);
    dst->val[1] = vsubq_s16(src0.val[1], src1.val[1]);
    dst->val[2] = vsubq_s16(src0.val[2], src1.val[2]);
}
template <>
inline void vsuq_s16<4, int16x8x4_t>(int16x8x4_t *dst, int16x8x4_t src0, int16x8x4_t src1)
{
    dst->val[0] = vsubq_s16(src0.val[0], src1.val[0]);
    dst->val[1] = vsubq_s16(src0.val[1], src1.val[1]);
    dst->val[2] = vsubq_s16(src0.val[2], src1.val[2]);
    dst->val[3] = vsubq_s16(src0.val[3], src1.val[3]);
}

template <>
inline void vadd_u8_f32<1, float32x4_t>(float32x4_t *dst, float32x4_t src)
{
    *dst = vaddq_f32(*dst, src);
}
template <>
inline void vadd_u8_f32<3, float32x4x3_t>(float32x4x3_t *dst, float32x4x3_t src)
{
    dst->val[0] = vaddq_f32(dst->val[0], src.val[0]);
    dst->val[1] = vaddq_f32(dst->val[1], src.val[1]);
    dst->val[2] = vaddq_f32(dst->val[2], src.val[2]);
}
template <>
inline void vadd_u8_f32<4, float32x4x4_t>(float32x4x4_t *dst, float32x4x4_t src)
{
    dst->val[0] = vaddq_f32(dst->val[0], src.val[0]);
    dst->val[1] = vaddq_f32(dst->val[1], src.val[1]);
    dst->val[2] = vaddq_f32(dst->val[2], src.val[2]);
    dst->val[3] = vaddq_f32(dst->val[3], src.val[3]);
}

template <int nc, typename T>
inline void vadd_u8_f32(T *dst, const T src0, const T src1);
template <>
inline void vadd_u8_f32<1, float32x4_t>(float32x4_t *dst, float32x4_t src0, float32x4_t src1)
{
    *dst = vaddq_f32(src0, src1);
}
template <>
inline void vadd_u8_f32<3, float32x4x3_t>(float32x4x3_t *dst, float32x4x3_t src0, float32x4x3_t src1)
{
    dst->val[0] = vaddq_f32(src0.val[0], src1.val[0]);
    dst->val[1] = vaddq_f32(src0.val[1], src1.val[1]);
    dst->val[2] = vaddq_f32(src0.val[2], src1.val[2]);
}
template <>
inline void vadd_u8_f32<4, float32x4x4_t>(float32x4x4_t *dst, float32x4x4_t src0, float32x4x4_t src1)
{
    dst->val[0] = vaddq_f32(src0.val[0], src1.val[0]);
    dst->val[1] = vaddq_f32(src0.val[1], src1.val[1]);
    dst->val[2] = vaddq_f32(src0.val[2], src1.val[2]);
    dst->val[3] = vaddq_f32(src0.val[3], src1.val[3]);
}

template <int nc, typename T>
inline void vmul_u8_f32(T *dst, T src);
template <>
inline void vmul_u8_f32<1, float32x4_t>(float32x4_t *dst, float32x4_t src)
{
    *dst = vmulq_f32(*dst, src);
}
template <>
inline void vmul_u8_f32<3, float32x4x3_t>(float32x4x3_t *dst, float32x4x3_t src)
{
    dst->val[0] = vmulq_f32(dst->val[0], src.val[0]);
    dst->val[1] = vmulq_f32(dst->val[1], src.val[1]);
    dst->val[2] = vmulq_f32(dst->val[2], src.val[2]);
}
template <>
inline void vmul_u8_f32<4, float32x4x4_t>(float32x4x4_t *dst, float32x4x4_t src)
{
    dst->val[0] = vmulq_f32(dst->val[0], src.val[0]);
    dst->val[1] = vmulq_f32(dst->val[1], src.val[1]);
    dst->val[2] = vmulq_f32(dst->val[2], src.val[2]);
    dst->val[3] = vmulq_f32(dst->val[3], src.val[3]);
}

template <int nc, typename T2, typename T>
inline void vset_u8_f32(T *ptr, T2 val);
template <>
inline void vset_u8_f32<1, float, float32x4_t>(float32x4_t *vec, float val)
{
    *vec = vdupq_n_f32(val);
}
template <>
inline void vset_u8_f32<3, float, float32x4x3_t>(float32x4x3_t *vec, float val)
{
    vec->val[0] = vdupq_n_f32(val);
    vec->val[1] = vdupq_n_f32(val);
    vec->val[2] = vdupq_n_f32(val);
}
template <>
inline void vset_u8_f32<4, float, float32x4x4_t>(float32x4x4_t *vec, float val)
{
    vec->val[0] = vdupq_n_f32(val);
    vec->val[1] = vdupq_n_f32(val);
    vec->val[2] = vdupq_n_f32(val);
    vec->val[3] = vdupq_n_f32(val);
}

template <>
inline void changeDT_u8x8_s16x8<1, uint8x8_t, int16x8_t>(const uint8x8_t src, int16x8_t &dst)
{
    uint16x8_t vec_16x8_val = vmovl_u8(src);
    dst = (int16x8_t)vec_16x8_val;
}
template <>
inline void changeDT_u8x8_s16x8<3, uint8x8x3_t, int16x8x3_t>(const uint8x8x3_t src, int16x8x3_t &dst)
{
    uint16x8_t vec_16x8_val = vmovl_u8(src.val[0]);
    dst.val[0] = (int16x8_t)vec_16x8_val;

    vec_16x8_val = vmovl_u8(src.val[1]);
    dst.val[1] = (int16x8_t)vec_16x8_val;

    vec_16x8_val = vmovl_u8(src.val[2]);
    dst.val[2] = (int16x8_t)vec_16x8_val;
}

template <>
inline void changeDT_u8x8_s16x8<4, uint8x8x4_t, int16x8x4_t>(const uint8x8x4_t src, int16x8x4_t &dst)
{
    uint16x8_t vec_16x8_val = vmovl_u8(src.val[0]);
    dst.val[0] = (int16x8_t)vec_16x8_val;

    vec_16x8_val = vmovl_u8(src.val[1]);
    dst.val[1] = (int16x8_t)vec_16x8_val;

    vec_16x8_val = vmovl_u8(src.val[2]);
    dst.val[2] = (int16x8_t)vec_16x8_val;

    vec_16x8_val = vmovl_u8(src.val[3]);
    dst.val[3] = (int16x8_t)vec_16x8_val;
}

template <>
inline void changeDT_u8x8_f32x4<1, uint8x8_t, float32x4_t>(const uint8x8_t src, float32x4_t &dstHigh, float32x4_t &dstLow)
{
    uint16x8_t vec_16x8_val = vmovl_u8(src);
    uint16x4_t vec_16x4_low = vget_low_u16(vec_16x8_val);
    uint16x4_t vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));
}
template <>
inline void changeDT_u8x8_f32x4<3, uint8x8x3_t, float32x4x3_t>(const uint8x8x3_t src, float32x4x3_t &dstHigh, float32x4x3_t &dstLow)
{
    uint16x8_t vec_16x8_val = vmovl_u8(src.val[0]);
    uint16x4_t vec_16x4_low = vget_low_u16(vec_16x8_val);
    uint16x4_t vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh.val[0] = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow.val[0] = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));

    vec_16x8_val = vmovl_u8(src.val[1]);
    vec_16x4_low = vget_low_u16(vec_16x8_val);
    vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh.val[1] = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow.val[1] = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));

    vec_16x8_val = vmovl_u8(src.val[2]);
    vec_16x4_low = vget_low_u16(vec_16x8_val);
    vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh.val[2] = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow.val[2] = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));
}

template <>
inline void changeDT_u8x8_f32x4<4, uint8x8x4_t, float32x4x4_t>(const uint8x8x4_t src, float32x4x4_t &dstHigh, float32x4x4_t &dstLow)
{
    uint16x8_t vec_16x8_val = vmovl_u8(src.val[0]);
    uint16x4_t vec_16x4_low = vget_low_u16(vec_16x8_val);
    uint16x4_t vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh.val[0] = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow.val[0] = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));

    vec_16x8_val = vmovl_u8(src.val[1]);
    vec_16x4_low = vget_low_u16(vec_16x8_val);
    vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh.val[1] = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow.val[1] = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));

    vec_16x8_val = vmovl_u8(src.val[2]);
    vec_16x4_low = vget_low_u16(vec_16x8_val);
    vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh.val[2] = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow.val[2] = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));

    vec_16x8_val = vmovl_u8(src.val[3]);
    vec_16x4_low = vget_low_u16(vec_16x8_val);
    vec_16x4_high = vget_high_u16(vec_16x8_val);
    dstHigh.val[3] = vcvtq_f32_u32(vmovl_u16(vec_16x4_high));
    dstLow.val[3] = vcvtq_f32_u32(vmovl_u16(vec_16x4_low));
}

template <>
inline void changeDT_f32x4_u8x8<1, float32x4_t, uint8x8_t>(const float32x4_t srcHigh, const float32x4_t srcLow, uint8x8_t &dst)
{
    uint16x4_t vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh, vdupq_n_f32(0.5f))));
    uint16x4_t vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow, vdupq_n_f32(0.5f))));
    uint16x8_t vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst = vqmovn_u16(vec_16x8_val);
}
template <>
inline void changeDT_f32x4_u8x8<3, float32x4x3_t, uint8x8x3_t>(const float32x4x3_t srcHigh, const float32x4x3_t srcLow, uint8x8x3_t &dst)
{
    uint16x4_t vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh.val[0], vdupq_n_f32(0.5f))));
    uint16x4_t vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow.val[0], vdupq_n_f32(0.5f))));
    uint16x8_t vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst.val[0] = vqmovn_u16(vec_16x8_val);

    vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh.val[1], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow.val[1], vdupq_n_f32(0.5f))));
    vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst.val[1] = vqmovn_u16(vec_16x8_val);

    vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh.val[2], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow.val[2], vdupq_n_f32(0.5f))));
    vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst.val[2] = vqmovn_u16(vec_16x8_val);
}
template <>
inline void changeDT_f32x4_u8x8<4, float32x4x4_t, uint8x8x4_t>(const float32x4x4_t srcHigh, const float32x4x4_t srcLow, uint8x8x4_t &dst)
{
    uint16x4_t vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh.val[0], vdupq_n_f32(0.5f))));
    uint16x4_t vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow.val[0], vdupq_n_f32(0.5f))));
    uint16x8_t vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst.val[0] = vqmovn_u16(vec_16x8_val);

    vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh.val[1], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow.val[1], vdupq_n_f32(0.5f))));
    vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst.val[1] = vqmovn_u16(vec_16x8_val);

    vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh.val[2], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow.val[2], vdupq_n_f32(0.5f))));
    vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst.val[2] = vqmovn_u16(vec_16x8_val);

    vec_16x4_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcHigh.val[3], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_u32(vcvtq_u32_f32(vaddq_f32(srcLow.val[3], vdupq_n_f32(0.5f))));
    vec_16x8_val = vcombine_u16(vec_16x4_low, vec_16x4_high);
    dst.val[3] = vqmovn_u16(vec_16x8_val);
}

template <>
inline void changeDT_f32x4_s16x8<1, float32x4_t, int16x8_t>(const float32x4_t srcHigh, const float32x4_t srcLow, int16x8_t &dst)
{
    int16x4_t vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh, vdupq_n_f32(0.5f))));
    int16x4_t vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow, vdupq_n_f32(0.5f))));
    dst = vcombine_s16(vec_16x4_low, vec_16x4_high);
}
template <>
inline void changeDT_f32x4_s16x8<3, float32x4x3_t, int16x8x3_t>(const float32x4x3_t srcHigh, const float32x4x3_t srcLow, int16x8x3_t &dst)
{
    int16x4_t vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh.val[0], vdupq_n_f32(0.5f))));
    int16x4_t vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow.val[0], vdupq_n_f32(0.5f))));
    dst.val[0] = vcombine_s16(vec_16x4_low, vec_16x4_high);

    vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh.val[1], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow.val[1], vdupq_n_f32(0.5f))));
    dst.val[1] = vcombine_s16(vec_16x4_low, vec_16x4_high);

    vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh.val[2], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow.val[2], vdupq_n_f32(0.5f))));
    dst.val[2] = vcombine_s16(vec_16x4_low, vec_16x4_high);
}
template <>
inline void changeDT_f32x4_s16x8<4, float32x4x4_t, int16x8x4_t>(const float32x4x4_t srcHigh, const float32x4x4_t srcLow, int16x8x4_t &dst)
{
    int16x4_t vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh.val[0], vdupq_n_f32(0.5f))));
    int16x4_t vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow.val[0], vdupq_n_f32(0.5f))));
    dst.val[0] = vcombine_s16(vec_16x4_low, vec_16x4_high);

    vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh.val[1], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow.val[1], vdupq_n_f32(0.5f))));
    dst.val[1] = vcombine_s16(vec_16x4_low, vec_16x4_high);

    vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh.val[2], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow.val[2], vdupq_n_f32(0.5f))));
    dst.val[2] = vcombine_s16(vec_16x4_low, vec_16x4_high);

    vec_16x4_high = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcHigh.val[3], vdupq_n_f32(0.5f))));
    vec_16x4_low = vmovn_s32(vcvtq_s32_f32(vaddq_f32(srcLow.val[3], vdupq_n_f32(0.5f))));
    dst.val[3] = vcombine_s16(vec_16x4_low, vec_16x4_high);
}

} // namespace aarch64
} // namespace cv
} // namespace ppl

#endif //! __ST_HPC_PPL_CV_ARM_TYPETRAITS_H_
