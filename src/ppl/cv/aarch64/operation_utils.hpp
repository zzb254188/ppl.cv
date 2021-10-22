// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for mulitional information
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
#ifndef OP_UTILS_HPP
#define OP_UTILS_HPP

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <stdint.h>

namespace ppl {
namespace cv {
namespace aarch64 {

    template<typename T>
        static inline T min(T x, T y){ 
            return (x<y)?x:y;
        }   

    template<typename T>
        static inline T max(T x, T y){ 
            return (x>y)?x:y;
        } 

    static inline int round(int v){ 
        return v;
    }   
    // !!! ONLY AVAILABLE WHEN v >= 0
    static inline int round(float v){ 
        return (int)(v+0.5f);
    }   
    // !!! ONLY AVAILABLE WHEN v >= 0
    static inline int round(double v){ 
        return (int)(v+0.5);
    }   

    template<typename _Tp> static inline _Tp saturate_cast(uint8_t v) { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(int8_t v) { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(uint16_t v) { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(int16_t v) { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(unsigned v) { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(int v) { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(float v) { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(double v) { return _Tp(v); }

    template<> inline uint8_t saturate_cast<uint8_t>(int8_t v)
    { return (uint8_t)max((int)v, 0); }
    template<> inline uint8_t saturate_cast<uint8_t>(uint16_t v)
    { return (uint8_t)min((unsigned)v, (unsigned)UCHAR_MAX); }
    template<> inline uint8_t saturate_cast<uint8_t>(int v)
    { return (uint8_t)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
    template<> inline uint8_t saturate_cast<uint8_t>(int16_t v)
    {return v >= UCHAR_MAX ? UCHAR_MAX : v < 0 ? 0 : v;}
    template<> inline uint8_t saturate_cast<uint8_t>(unsigned v)
    { return (uint8_t)min(v, (unsigned)UCHAR_MAX); }
    template<> inline uint8_t saturate_cast<uint8_t>(float v)
    { int iv = round(v); return saturate_cast<uint8_t>(iv); }
    template<> inline uint8_t saturate_cast<uint8_t>(double v)
    { int iv = round(v); return saturate_cast<uint8_t>(iv); }

    template<> inline int8_t saturate_cast<int8_t>(uint8_t v)
    { return (int8_t)min((int)v, SCHAR_MAX); }
    template<> inline int8_t saturate_cast<int8_t>(uint16_t v)
    { return (int8_t)min((unsigned)v, (unsigned)SCHAR_MAX); }
    template<> inline int8_t saturate_cast<int8_t>(int v)
    {
        return (int8_t)((unsigned)(v-SCHAR_MIN) <= (unsigned)UCHAR_MAX ?
                v : v > 0 ? SCHAR_MAX : SCHAR_MIN);
    }
    template<> inline int8_t saturate_cast<int8_t>(int16_t v)
    { return saturate_cast<int8_t>((int)v); }
    template<> inline int8_t saturate_cast<int8_t>(unsigned v)
    { return (int8_t)min(v, (unsigned)SCHAR_MAX); }

    template<> inline int8_t saturate_cast<int8_t>(float v)
    { int iv = round(v); return saturate_cast<int8_t>(iv); }
    template<> inline int8_t saturate_cast<int8_t>(double v)
    { int iv = round(v); return saturate_cast<int8_t>(iv); }

    template<> inline uint16_t saturate_cast<uint16_t>(int8_t v)
    { return (uint16_t)max((int)v, 0); }
    template<> inline uint16_t saturate_cast<uint16_t>(int16_t v)
    { return (uint16_t)max((int)v, 0); }
    template<> inline uint16_t saturate_cast<uint16_t>(int v)
    { return (uint16_t)((unsigned)v <= (unsigned)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
    template<> inline uint16_t saturate_cast<uint16_t>(unsigned v)
    { return (uint16_t)min(v, (unsigned)USHRT_MAX); }
    template<> inline uint16_t saturate_cast<uint16_t>(float v)
    { int iv = round(v); return saturate_cast<uint16_t>(iv); }
    template<> inline uint16_t saturate_cast<uint16_t>(double v)
    { int iv = round(v); return saturate_cast<uint16_t>(iv); }

    template<> inline int16_t saturate_cast<int16_t>(uint16_t v)
    { return (int16_t)min((int)v, SHRT_MAX); }
    template<> inline int16_t saturate_cast<int16_t>(int v)
    {
        return (int16_t)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ?
                v : v > 0 ? SHRT_MAX : SHRT_MIN);
    }
    template<> inline int16_t saturate_cast<int16_t>(unsigned v)
    { return (int16_t)min(v, (unsigned)SHRT_MAX); }
    template<> inline int16_t saturate_cast<int16_t>(float v)
    { int iv = round(v); return saturate_cast<int16_t>(iv); }
    template<> inline int16_t saturate_cast<int16_t>(double v)
    { int iv = round(v); return saturate_cast<int16_t>(iv); }

    template<> inline int saturate_cast<int>(float v) { return round(v); }
    template<> inline int saturate_cast<int>(double v) { return round(v); }

    // we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
    template<> inline unsigned saturate_cast<unsigned>(float v){ return round(v); }
    template<> inline unsigned saturate_cast<unsigned>(double v) { return round(v); }
}
}
}
#endif
