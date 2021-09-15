#ifndef __ST_HPC_PPL_CV_AARCH64_COMMON_H_
#define __ST_HPC_PPL_CV_AARCH64_COMMON_H_

namespace ppl {
namespace cv {
namespace aarch64 {

inline void prefetch(const void *ptr, size_t offset = 1024)
{
#if defined __GNUC__
    __builtin_prefetch(reinterpret_cast<const char*>(ptr) + offset);
#elif defined _MSC_VER && defined CAROTENE_NEON
    __prefetch(reinterpret_cast<const char*>(ptr) + offset);
#else
    (void)ptr;
    (void)offset;
#endif
}

}}}//ppl::cv::aarch64
#endif