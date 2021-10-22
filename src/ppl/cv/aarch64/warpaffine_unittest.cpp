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

#include "ppl/cv/aarch64/warpaffine.h"
#include "ppl/cv/aarch64/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int val>
static void randomRangeData(T *data, const size_t num,int maxNum =255){
    size_t tmp;

    for(size_t i = 0; i < num; i++){
        tmp = rand()% maxNum;
        data[i] = (T) ( (float)tmp/(float)val );
    }
}

template <typename T, int32_t nc>
class WarpAffine : public ::testing::TestWithParam<std::tuple<int, int, int, float>> {
public:
    using WarpAffineParam = std::tuple<int, int, int, float>;
    WarpAffine()
    {
    }

    ~WarpAffine()
    {
    }

    void Linearapply(const WarpAffineParam &param) {
        int width = std::get<0>(param);
        int height = std::get<1>(param);
        int borderType = std::get<2>(param);
        float diff = std::get<3>(param);

        std::unique_ptr<T[]> src(new T[width * height * nc]);
        std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
        std::unique_ptr<T[]> dst(new T[width * height * nc]);
        std::unique_ptr<float[]> affineMatrix0(new float[6]);
        std::unique_ptr<float[]> affineMatrix(new float[6]);
        ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
        randomRangeData<float, 128>(affineMatrix0.get(), 6);
        ppl::cv::debug::randomFill<T>(dst_ref.get(), width * height * nc, 0, 255);
        memcpy(dst.get(), dst_ref.get(), width * height * nc * sizeof(T));
        memcpy(affineMatrix.get(), affineMatrix0.get(), 6 * sizeof(float));    
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);
        cv::Mat affineMatrix_opencv(2, 3, CV_32FC1, affineMatrix0.get());

        T border_value;
        ppl::cv::debug::randomFill<T>(&border_value, 1, 0, 255);

        cv::Scalar borderValue = {border_value, border_value, border_value, border_value};
        cv::warpAffine(src_opencv, dst_opencv, affineMatrix_opencv, dst_opencv.size(), 17/*CV_WARP_INVERSE_MAP + CV_INTER_LINEAR*/, 
            borderType, borderValue);

        ppl::cv::aarch64::WarpAffineLinear<T, nc>(
            height,
            width,
            width * nc,
            src.get(),
            height,
            width,
            width * nc,
            dst.get(),
            affineMatrix.get(),
            (ppl::cv::BorderType)borderType,
            border_value);

        checkResult<T, nc>(
            dst_ref.get(),
            dst.get(),
            height,
            width,
            width * nc,
            width * nc,
            diff);
    };
};

#define R1(name, t, c, diff)\
    using name = WarpAffine<t, c>;\
    TEST_P(name, abc)\
    {\
        this->Linearapply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
        ::testing::Combine(::testing::Values(2, 2), \
                           ::testing::Values(4, 4), \
                           ::testing::Values(ppl::cv::BORDER_TYPE_CONSTANT),\
                           ::testing::Values(diff)));

R1(WarpAffineLinear_f8c1, float, 1, 1e-2)
R1(WarpAffineLinear_f8c3, float, 3, 1e-2)
R1(WarpAffineLinear_f8c4, float, 4, 1e-2)

R1(WarpAffineLinear_u8c1, uint8_t, 1, 1.01)
R1(WarpAffineLinear_u8c2, uint8_t, 2, 1.01)
R1(WarpAffineLinear_u8c3, uint8_t, 3, 1.01)
R1(WarpAffineLinear_u8c4, uint8_t, 4, 1.01)
