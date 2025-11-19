#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <cmath>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {
namespace detail {
#if defined(__AVX2__)
inline __m256 madd_ps(__m256 acc, __m256 a, __m256 b) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, acc);
#else
    return _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#endif
}

inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline void expand_u8x16_to_ps(__m128i src, __m256& lo, __m256& hi) {
    __m256i lo_i = _mm256_cvtepu8_epi32(src);
    __m128i hi_bytes = _mm_srli_si128(src, 8);
    __m256i hi_i = _mm256_cvtepu8_epi32(hi_bytes);
    lo = _mm256_cvtepi32_ps(lo_i);
    hi = _mm256_cvtepi32_ps(hi_i);
}

inline __m256 expand_u8x8_to_ps(__m128i src) {
    __m256i wide = _mm256_cvtepu8_epi32(src);
    return _mm256_cvtepi32_ps(wide);
}

inline __m256i mask_from_u8(uint8_t mask) {
    return _mm256_set_epi32(
        (mask & 0x80) ? -1 : 0,
        (mask & 0x40) ? -1 : 0,
        (mask & 0x20) ? -1 : 0,
        (mask & 0x10) ? -1 : 0,
        (mask & 0x08) ? -1 : 0,
        (mask & 0x04) ? -1 : 0,
        (mask & 0x02) ? -1 : 0,
        (mask & 0x01) ? -1 : 0
    );
}

inline __m256 apply_sign_flip(__m256 values, uint8_t mask_bits) {
    const __m256 sign_flip = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    __m256i mask_vec = mask_from_u8(mask_bits);
    __m256 flip = _mm256_and_ps(sign_flip, _mm256_castsi256_ps(mask_vec));
    return _mm256_xor_ps(values, flip);
}
#endif  // defined(__AVX2__)
inline uint32_t extract_interleaved_bit(uint64_t bits, size_t index) {
    size_t block = index / 8;
    size_t lane = index % 8;
    size_t bit_index = block + (lane * 8);
    return static_cast<uint32_t>((bits >> bit_index) & 0x1ULL);
}
}  // namespace detail
// Small helpers for runtime verification when both AVX512F and AVX2 are available.
inline bool float_eq(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

inline bool compare_float_arrays(const float* a, const float* b, size_t n, float tol = 1e-4f) {
    for (size_t i = 0; i < n; ++i) {
        if (!float_eq(a[i], b[i], tol)) return false;
    }
    return true;
}
namespace scalar_impl {
template <typename T>
void scalar_quantize_normal(
    T* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    float one_over_delta = 1.0F / delta;

    ConstRowMajorArrayMap<float> v0(vec0, 1, static_cast<long>(dim));
    RowMajorArrayMap<T> res(result, 1, dim);

    // round to nearest integer, then cast to integer
    res = ((v0 - lo) * one_over_delta).round().template cast<T>();
}

template <typename T>
void scalar_quantize_optimized(
    T* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    scalar_quantize_normal(result, vec0, dim, lo, delta);
}

template <>
inline void scalar_quantize_optimized<uint8_t>(
    uint8_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    auto lo512 = _mm512_set1_ps(lo);
    auto od512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        auto cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512);
        auto i8 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(cur));
        _mm_storeu_epi8(&result[i], i8);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
#elif defined(__AVX2__)
    size_t mul16 = dim & ~static_cast<size_t>(15);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    const __m256 lo256 = _mm256_set1_ps(lo);
    const __m256 od256 = _mm256_set1_ps(one_over_delta);

    for (; i < mul16; i += 16) {
        __m256 cur0 = _mm256_loadu_ps(&vec0[i]);
        __m256 cur1 = _mm256_loadu_ps(&vec0[i + 8]);

        cur0 = _mm256_mul_ps(_mm256_sub_ps(cur0, lo256), od256);
        cur1 = _mm256_mul_ps(_mm256_sub_ps(cur1, lo256), od256);

        __m256i ints0 = _mm256_cvtps_epi32(cur0);
        __m256i ints1 = _mm256_cvtps_epi32(cur1);

        __m256i packed16 = _mm256_packs_epi32(ints0, ints1);
        packed16 = _mm256_permute4x64_epi64(packed16, 0b11011000);

        __m128i low16 = _mm256_castsi256_si128(packed16);
        __m128i high16 = _mm256_extracti128_si256(packed16, 1);
        __m128i packed8 = _mm_packus_epi16(low16, high16);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&result[i]), packed8);
    }

    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
#else
    scalar_quantize_normal(result, vec0, dim, lo, delta);
#endif
}

template <>
inline void scalar_quantize_optimized<uint16_t>(
    uint16_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    auto lo512 = _mm512_set1_ps(lo);
    auto ow512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        auto cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), ow512);
        auto i16 = _mm512_cvtepi32_epi16(_mm512_cvtps_epi32(cur));
        _mm256_storeu_epi16(&result[i], i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
#elif defined(__AVX2__)
    size_t mul8 = dim - (dim & 0b111);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    auto lo256 = _mm256_set1_ps(lo);
    auto ow256 = _mm256_set1_ps(one_over_delta);
    for (; i < mul8; i += 8) {
        auto cur = _mm256_loadu_ps(&vec0[i]);
        cur = _mm256_mul_ps(_mm256_sub_ps(cur, lo256), ow256);
        auto i16 = _mm256_cvtepi32_epi16(_mm256_cvtps_epi32(cur));
        _mm_storeu_epi16(&result[i], i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
#else
    scalar_quantize_normal(result, vec0, dim, lo, delta);
#endif
}
}  // namespace scalar_impl

template <typename T>
inline void vec_rescale(T* data, size_t dim, T val) {
    RowMajorArrayMap<T> data_arr(data, 1, dim);
    data_arr *= val;
}

template <typename T>
inline T euclidean_sqr(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return (v0 - v1).dot(v0 - v1);
}

template <typename T>
inline T dot_product_dis(
    const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim
) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return 1 - v0.dot(v1);
}

template <typename T>
inline T l2norm_sqr(const T* __restrict__ vec0, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    return v0.dot(v0);
}

template <typename T>
inline T dot_product(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return v0.dot(v1);
}

template <typename T>
inline T normalize_vec(
    const T* __restrict__ vec, const T* __restrict__ centroid, T* res, T dist2c, size_t dim
) {
    RowMajorArrayMap<T> r(res, 1, dim);
    if (dist2c > 1e-5) {
        ConstRowMajorArrayMap<T> v(vec, 1, dim);
        ConstRowMajorArrayMap<T> c(centroid, 1, dim);
        r = (v - c) * (1 / dist2c);
        return r.sum();
    }
    T value = 1.0 / std::sqrt(static_cast<T>(dim));
    r = value;
    return static_cast<T>(dim) * value;
}

// pack 0/1 data to usigned integer
template <typename T>
inline void pack_binary(
    const int* __restrict__ binary_code, T* __restrict__ compact_code, size_t length
) {
    constexpr size_t kTypeBits = sizeof(T) * 8;

    for (size_t i = 0; i < length; i += kTypeBits) {
        T cur = 0;
        for (size_t j = 0; j < kTypeBits; ++j) {
            cur |= (static_cast<T>(binary_code[i + j]) << (kTypeBits - 1 - j));
        }
        *compact_code = cur;
        ++compact_code;
    }
}

template <typename T>
inline void data_range(const T* __restrict__ vec0, size_t dim, T& lo, T& hi) {
    ConstRowMajorArrayMap<T> v0(vec0, 1, dim);
    lo = v0.minCoeff();
    hi = v0.maxCoeff();
}

template <typename T, typename TD>
void scalar_quantize(
    T* __restrict__ result, const TD* __restrict__ vec0, size_t dim, TD lo, TD delta
) {
    assert_integral<T>();
    scalar_impl::scalar_quantize_optimized(result, vec0, dim, lo, delta);
}

template <typename T>
inline std::vector<T> compute_centroid(
    const T* data, size_t num_points, size_t dim, size_t num_threads
) {
    omp_set_num_threads(static_cast<int>(num_threads));
    std::vector<std::vector<T>> all_results(num_threads, std::vector<T>(dim, 0));

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        std::vector<T>& cur_results = all_results[tid];
        const T* cur_data = data + (dim * i);
        for (size_t k = 0; k < dim; ++k) {
            cur_results[k] += cur_data[k];
        }
    }

    std::vector<T> centroid(dim, 0);
    for (auto& one_res : all_results) {
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] += one_res[i];
        }
    }
    T inv_num_points = 1 / static_cast<T>(num_points);

    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = centroid[i] * inv_num_points;
    }

    return centroid;
}

template <typename T>
inline PID exact_nn(
    const T* data,
    const T* query,
    size_t num_points,
    size_t dim,
    size_t num_threads,
    T (*dist_func)(const T*, const T*, size_t)
) {
    std::vector<AnnCandidate<T, PID>> best_entries(num_threads);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        AnnCandidate<T, PID>& cur_entry = best_entries[tid];
        const T* cur_data = data + (dim * i);

        T distance = dist_func(cur_data, query, dim);
        if (distance < cur_entry.distance) {
            cur_entry.id = static_cast<PID>(i);
            cur_entry.distance = distance;
        }
    }

    PID nearest_neighbor = 0;
    T min_dist = std::numeric_limits<T>::max();
    for (auto& candi : best_entries) {
        if (candi.distance < min_dist) {
            nearest_neighbor = candi.id;
            min_dist = candi.distance;
        }
    }
    return nearest_neighbor;
}

namespace excode_ipimpl {
// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
inline float ip16_fxu1_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    float result = 0.0F;
    __m512 sum = _mm512_setzero_ps();

    const float* q_ptr = query;
    const uint8_t* code_ptr = compact_code;
    for (size_t i = 0; i < dim; i += 16) {
        __mmask16 mask = *reinterpret_cast<const __mmask16*>(code_ptr);
        __m512 q = _mm512_loadu_ps(q_ptr);
        sum = _mm512_add_ps(_mm512_maskz_mov_ps(mask, q), sum);
        code_ptr += 2;
        q_ptr += 16;
    }
    result = _mm512_reduce_add_ps(sum);
    return result;
#elif defined(__AVX2__)
    float total = 0.0F;
    size_t processed = 0;
    const uint8_t* code_ptr = compact_code;

    while (processed + 16 <= dim) {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(code_ptr);
        alignas(32) float decoded[16];
        for (int j = 0; j < 16; ++j) {
            decoded[j] = static_cast<float>((bits >> j) & 0x1U);
        }

        __m256 q0 = _mm256_loadu_ps(query + processed);
        __m256 q1 = _mm256_loadu_ps(query + processed + 8);
        __m256 d0 = _mm256_loadu_ps(decoded);
        __m256 d1 = _mm256_loadu_ps(decoded + 8);
        __m256 prod0 = _mm256_mul_ps(q0, d0);
        __m256 prod1 = _mm256_mul_ps(q1, d1);
        total += detail::hsum_ps(_mm256_add_ps(prod0, prod1));

        processed += 16;
        code_ptr += 2;
    }

    for (; processed < dim; ++processed) {
        size_t byte_idx = processed / 8;
        size_t bit_idx = processed % 8;
        if ((compact_code[byte_idx] >> bit_idx) & 0x1U) {
            total += query[processed];
        }
    }

    return total;
#else
    float result = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        if ((compact_code[byte_idx] >> bit_idx) & 0x1U) {
            result += query[i];
        }
    }
    return result;
#endif
}

inline float ip16_fxu2_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    float result = 0.0F;
    __m512 sum = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0b00000011);
    const uint8_t* code_ptr = compact_code;

    for (size_t i = 0; i < dim; i += 16) {
        int32_t compact = *reinterpret_cast<const int32_t*>(code_ptr);
        __m128i code = _mm_set_epi32(compact >> 6, compact >> 4, compact >> 2, compact);
        code = _mm_and_si128(code, mask);

        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(code));
        __m512 q = _mm512_loadu_ps(query + i);
        sum = _mm512_fmadd_ps(cf, q, sum);
        code_ptr += 4;
    }
    result = _mm512_reduce_add_ps(sum);
    return result;
#elif defined(__AVX2__)
    __m256 acc_lo = _mm256_setzero_ps();
    __m256 acc_hi = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0b00000011);

    size_t processed = 0;
    const uint8_t* code_ptr = compact_code;
    while (processed + 16 <= dim) {
        int32_t compact = *reinterpret_cast<const int32_t*>(code_ptr);
        __m128i code = _mm_set_epi32(compact >> 6, compact >> 4, compact >> 2, compact);
        code = _mm_and_si128(code, mask);

        __m256 cf_lo;
        __m256 cf_hi;
        detail::expand_u8x16_to_ps(code, cf_lo, cf_hi);

        acc_lo = detail::madd_ps(acc_lo, cf_lo, _mm256_loadu_ps(query + processed));
        acc_hi = detail::madd_ps(acc_hi, cf_hi, _mm256_loadu_ps(query + processed + 8));

        processed += 16;
        code_ptr += 4;
    }

    float result = detail::hsum_ps(_mm256_add_ps(acc_lo, acc_hi));

    for (; processed < dim; ++processed) {
        size_t code_index = processed / 16;
        size_t bit_offset = (processed % 16) * 2;
        uint32_t packed = *reinterpret_cast<const uint32_t*>(compact_code + code_index * 4);
        uint32_t value = (packed >> bit_offset) & 0x3U;
        result += query[processed] * static_cast<float>(value);
    }
    return result;
#else
    float result = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        size_t code_index = i / 16;
        size_t bit_offset = (i % 16) * 2;
        uint32_t packed = *reinterpret_cast<const uint32_t*>(compact_code + code_index * 4);
        uint32_t value = (packed >> bit_offset) & 0x3U;
        result += query[i] * static_cast<float>(value);
    }
    return result;
#endif
}

inline float ip64_fxu3_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    const __m128i mask = _mm_set1_epi8(0b11);
    const __m128i top_mask = _mm_set1_epi8(0b100);
    const uint8_t* code_ptr = compact_code;

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code_ptr));
        code_ptr += 16;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(code_ptr);
        code_ptr += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact2, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact2, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact2, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact2, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
    }

    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    const __m128i mask = _mm_set1_epi8(0b11);
    const __m128i top_mask = _mm_set1_epi8(0b100);
    float result = 0.0F;

    for (size_t i = 0; i < dim; i += 64) {
        size_t block = i / 64;
        const uint8_t* base_ptr = compact_code + block * 24;
        __m128i compact2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_ptr));
        int64_t top_bit = *reinterpret_cast<const int64_t*>(base_ptr + 16);

        __m128i vec_00_to_15 = _mm_and_si128(compact2, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact2, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact2, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact2, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m256 cf_lo;
        __m256 cf_hi;

        detail::expand_u8x16_to_ps(vec_00_to_15, cf_lo, cf_hi);
        result += detail::hsum_ps(_mm256_mul_ps(cf_lo, _mm256_loadu_ps(query + i)));
        result += detail::hsum_ps(_mm256_mul_ps(cf_hi, _mm256_loadu_ps(query + i + 8)));

        detail::expand_u8x16_to_ps(vec_16_to_31, cf_lo, cf_hi);
        result += detail::hsum_ps(_mm256_mul_ps(cf_lo, _mm256_loadu_ps(query + i + 16)));
        result += detail::hsum_ps(_mm256_mul_ps(cf_hi, _mm256_loadu_ps(query + i + 24)));

        detail::expand_u8x16_to_ps(vec_32_to_47, cf_lo, cf_hi);
        result += detail::hsum_ps(_mm256_mul_ps(cf_lo, _mm256_loadu_ps(query + i + 32)));
        result += detail::hsum_ps(_mm256_mul_ps(cf_hi, _mm256_loadu_ps(query + i + 40)));

        detail::expand_u8x16_to_ps(vec_48_to_63, cf_lo, cf_hi);
        result += detail::hsum_ps(_mm256_mul_ps(cf_lo, _mm256_loadu_ps(query + i + 48)));
        result += detail::hsum_ps(_mm256_mul_ps(cf_hi, _mm256_loadu_ps(query + i + 56)));
    }
    return result;
#else
    float result = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        size_t block = i / 64;
        size_t offset = i % 64;
        const uint8_t* base_ptr = compact_code + block * 24;
        const uint8_t* two_bit = base_ptr;
        uint8_t packed = two_bit[offset / 4];
        int value = (packed >> ((offset % 4) * 2)) & 0x3;
        int64_t top_bits = *reinterpret_cast<const int64_t*>(base_ptr + 16);
        value |= ((top_bits >> offset) & 0x1) << 2;
        result += query[i] * static_cast<float>(value);
    }
    return result;
#endif
}

inline float ip16_fxu4_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    constexpr uint64_t kMask = 0x0f0f0f0f0f0f0f0fULL;
    __m512 sum = _mm512_setzero_ps();
    const uint8_t* code_ptr = compact_code;

    for (size_t i = 0; i < dim; i += 16) {
        uint64_t compact = *reinterpret_cast<const uint64_t*>(code_ptr);
        uint64_t code0 = compact & kMask;
        uint64_t code1 = (compact >> 4) & kMask;

        __m128i c8 = _mm_set_epi64x(static_cast<long long>(code1), static_cast<long long>(code0));
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));

        __m512 q = _mm512_loadu_ps(query + i);
        sum = _mm512_fmadd_ps(cf, q, sum);
        code_ptr += 8;
    }
    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    constexpr uint64_t kMask = 0x0f0f0f0f0f0f0f0fULL;
    float total = 0.0F;
    size_t processed = 0;
    const uint8_t* code_ptr = compact_code;

    while (processed + 16 <= dim) {
        uint64_t compact = *reinterpret_cast<const uint64_t*>(code_ptr);
        uint64_t code0 = compact & kMask;
        uint64_t code1 = (compact >> 4) & kMask;

        __m128i bytes = _mm_set_epi64x(static_cast<long long>(code1), static_cast<long long>(code0));
        __m256 vals_lo;
        __m256 vals_hi;
        detail::expand_u8x16_to_ps(bytes, vals_lo, vals_hi);

        __m256 q0 = _mm256_loadu_ps(query + processed);
        __m256 q1 = _mm256_loadu_ps(query + processed + 8);
        __m256 prod0 = _mm256_mul_ps(q0, vals_lo);
        __m256 prod1 = _mm256_mul_ps(q1, vals_hi);
        total += detail::hsum_ps(_mm256_add_ps(prod0, prod1));

        processed += 16;
        code_ptr += 8;
    }

    for (; processed < dim; ++processed) {
        size_t byte_idx = processed / 2;
        uint8_t packed = compact_code[byte_idx];
        uint8_t value = (processed % 2 == 0) ? (packed & 0x0FU) : (packed >> 4);
        total += query[processed] * static_cast<float>(value);
    }

    return total;
#else
    float result = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        size_t byte_idx = i / 2;
        uint8_t packed = compact_code[byte_idx];
        uint8_t value = (i % 2 == 0) ? (packed & 0x0FU) : (packed >> 4);
        result += query[i] * static_cast<float>(value);
    }
    return result;
#endif
}

inline float ip64_fxu5_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    const __m128i mask = _mm_set1_epi8(0b1111);
    const __m128i top_mask = _mm_set1_epi8(0b10000);
    const uint8_t* code_ptr = compact_code;

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact4_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code_ptr));
        __m128i compact4_2 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(code_ptr + 16));
        code_ptr += 32;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(code_ptr);
        code_ptr += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact4_1, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact4_1, 4), mask);
        __m128i vec_32_to_47 = _mm_and_si128(compact4_2, mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact4_2, 4), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
    }

    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    const __m128i mask = _mm_set1_epi8(0b1111);
    const __m128i top_mask = _mm_set1_epi8(0b10000);
    float total = 0.0F;
    size_t processed = 0;

    while (processed + 64 <= dim) {
        size_t block = processed / 64;
        const uint8_t* base_ptr = compact_code + block * 40;  // 32 + 8 bytes

        __m128i compact4_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_ptr));
        __m128i compact4_2 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_ptr + 16));
        int64_t top_bit = *reinterpret_cast<const int64_t*>(base_ptr + 32);

        __m128i vec_00_to_15 = _mm_and_si128(compact4_1, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact4_1, 4), mask);
        __m128i vec_32_to_47 = _mm_and_si128(compact4_2, mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact4_2, 4), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m256 vals_lo;
        __m256 vals_hi;

        detail::expand_u8x16_to_ps(vec_00_to_15, vals_lo, vals_hi);
        __m256 q0 = _mm256_loadu_ps(query + processed);
        __m256 q1 = _mm256_loadu_ps(query + processed + 8);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        detail::expand_u8x16_to_ps(vec_16_to_31, vals_lo, vals_hi);
        q0 = _mm256_loadu_ps(query + processed + 16);
        q1 = _mm256_loadu_ps(query + processed + 24);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        detail::expand_u8x16_to_ps(vec_32_to_47, vals_lo, vals_hi);
        q0 = _mm256_loadu_ps(query + processed + 32);
        q1 = _mm256_loadu_ps(query + processed + 40);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        detail::expand_u8x16_to_ps(vec_48_to_63, vals_lo, vals_hi);
        q0 = _mm256_loadu_ps(query + processed + 48);
        q1 = _mm256_loadu_ps(query + processed + 56);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        processed += 64;
    }

    for (; processed < dim; ++processed) {
        size_t block = processed / 64;
        size_t offset = processed % 64;
        const uint8_t* base_ptr = compact_code + block * 40;
        const uint8_t* nibble1 = base_ptr;
        const uint8_t* nibble2 = base_ptr + 16;
        int64_t top_bits = *reinterpret_cast<const int64_t*>(base_ptr + 32);

        uint8_t low;
        if (offset < 16) {
            low = nibble1[offset] & 0x0F;
        } else if (offset < 32) {
            low = static_cast<uint8_t>((nibble1[offset - 16] >> 4) & 0x0F);
        } else if (offset < 48) {
            low = nibble2[offset - 32] & 0x0F;
        } else {
            low = static_cast<uint8_t>((nibble2[offset - 48] >> 4) & 0x0F);
        }
        uint8_t top = static_cast<uint8_t>(detail::extract_interleaved_bit(top_bits, offset));
        uint8_t value = static_cast<uint8_t>(low | (top << 4));
        total += query[processed] * static_cast<float>(value);
    }

    return total;
#else
    float result = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        size_t block = i / 64;
        size_t offset = i % 64;
        const uint8_t* base_ptr = compact_code + block * 40;
        const uint8_t* nibble1 = base_ptr;
        const uint8_t* nibble2 = base_ptr + 16;
        int64_t top_bits = *reinterpret_cast<const int64_t*>(base_ptr + 32);

        uint8_t low;
        if (offset < 16) {
            low = nibble1[offset] & 0x0F;
        } else if (offset < 32) {
            low = static_cast<uint8_t>((nibble1[offset - 16] >> 4) & 0x0F);
        } else if (offset < 48) {
            low = nibble2[offset - 32] & 0x0F;
        } else {
            low = static_cast<uint8_t>((nibble2[offset - 48] >> 4) & 0x0F);
        }
        uint8_t top = static_cast<uint8_t>(detail::extract_interleaved_bit(top_bits, offset));
        uint8_t value = static_cast<uint8_t>(low | (top << 4));
        result += query[i] * static_cast<float>(value);
    }
    return result;
#endif
}

inline float ip16_fxu6_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    constexpr uint64_t kMask4 = 0x0f0f0f0f0f0f0f0fULL;
    const __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m512 sum = _mm512_setzero_ps();
    const uint8_t* code_ptr = compact_code;

    for (size_t i = 0; i < dim; i += 16) {
        uint64_t compact4 = *reinterpret_cast<const uint64_t*>(code_ptr);
        uint64_t code4_0 = compact4 & kMask4;
        uint64_t code4_1 = (compact4 >> 4) & kMask4;

        __m128i c4 = _mm_set_epi64x(static_cast<long long>(code4_1), static_cast<long long>(code4_0));
        code_ptr += 8;

        int32_t compact2 = *reinterpret_cast<const int32_t*>(code_ptr);

        __m128i c2 = _mm_set_epi32(compact2 >> 2, compact2, compact2 << 2, compact2 << 4);
        c2 = _mm_and_si128(c2, mask2);

        __m128i c6 = _mm_or_si128(c2, c4);

        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c6));

        __m512 q = _mm512_loadu_ps(query + i);
        sum = _mm512_fmadd_ps(cf, q, sum);
        code_ptr += 4;
    }
    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    constexpr uint64_t kMask4 = 0x0f0f0f0f0f0f0f0fULL;
    const __m128i mask2 = _mm_set1_epi8(0b00110000);
    float total = 0.0F;
    size_t processed = 0;

    while (processed + 16 <= dim) {
        size_t block = processed / 16;
        const uint8_t* base_ptr = compact_code + block * 12;  // 8 + 4 bytes

        uint64_t compact4 = *reinterpret_cast<const uint64_t*>(base_ptr);
        uint64_t code4_0 = compact4 & kMask4;
        uint64_t code4_1 = (compact4 >> 4) & kMask4;

        __m128i c4 = _mm_set_epi64x(static_cast<long long>(code4_1), static_cast<long long>(code4_0));

        int32_t compact2 = *reinterpret_cast<const int32_t*>(base_ptr + 8);
        __m128i c2 = _mm_set_epi32(compact2 >> 2, compact2, compact2 << 2, compact2 << 4);
        c2 = _mm_and_si128(c2, mask2);

        __m128i c6 = _mm_or_si128(c2, c4);

        __m256 vals_lo;
        __m256 vals_hi;
        detail::expand_u8x16_to_ps(c6, vals_lo, vals_hi);

        __m256 q0 = _mm256_loadu_ps(query + processed);
        __m256 q1 = _mm256_loadu_ps(query + processed + 8);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        processed += 16;
    }

    for (; processed < dim; ++processed) {
        size_t block = processed / 16;
        size_t offset = processed % 16;
        const uint8_t* base_ptr = compact_code + block * 12;
        const uint8_t* low_ptr = base_ptr;
        const uint8_t* high_ptr = base_ptr + 8;

        uint8_t low;
        if (offset < 8) {
            low = low_ptr[offset] & 0x0F;
        } else {
            low = static_cast<uint8_t>((low_ptr[offset - 8] >> 4) & 0x0F);
        }

        size_t group = offset % 4;
        size_t block_high = offset / 4;
        uint8_t high = static_cast<uint8_t>((high_ptr[group] >> (block_high * 2)) & 0x3U);
        uint8_t value = static_cast<uint8_t>(low | (high << 4));
        total += query[processed] * static_cast<float>(value);
    }

    return total;
#else
    float result = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        size_t block = i / 16;
        size_t offset = i % 16;
        const uint8_t* base_ptr = compact_code + block * 12;
        const uint8_t* low_ptr = base_ptr;
        const uint8_t* high_ptr = base_ptr + 8;

        uint8_t low;
        if (offset < 8) {
            low = low_ptr[offset] & 0x0F;
        } else {
            low = static_cast<uint8_t>((low_ptr[offset - 8] >> 4) & 0x0F);
        }

        size_t group = offset % 4;
        size_t block_high = offset / 4;
        uint8_t high = static_cast<uint8_t>((high_ptr[group] >> (block_high * 2)) & 0x3U);
        uint8_t value = static_cast<uint8_t>(low | (high << 4));
        result += query[i] * static_cast<float>(value);
    }
    return result;
#endif
}

inline float ip64_fxu7_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(0b11000000);
    const __m128i top_mask = _mm_set1_epi8(0b1000000);
    const uint8_t* code_ptr = compact_code;

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code_ptr));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code_ptr + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code_ptr + 32));
        code_ptr += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        int64_t top_bit = *reinterpret_cast<const int64_t*>(code_ptr);
        code_ptr += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
    }

    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(0b11000000);
    const __m128i top_mask = _mm_set1_epi8(0b1000000);
    float total = 0.0F;
    size_t processed = 0;

    while (processed + 64 <= dim) {
        size_t block = processed / 64;
        const uint8_t* base_ptr = compact_code + block * 56;

        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_ptr));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_ptr + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_ptr + 32));
        int64_t top_bit = *reinterpret_cast<const int64_t*>(base_ptr + 48);

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m256 vals_lo;
        __m256 vals_hi;

        detail::expand_u8x16_to_ps(vec_00_to_15, vals_lo, vals_hi);
        __m256 q0 = _mm256_loadu_ps(query + processed);
        __m256 q1 = _mm256_loadu_ps(query + processed + 8);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        detail::expand_u8x16_to_ps(vec_16_to_31, vals_lo, vals_hi);
        q0 = _mm256_loadu_ps(query + processed + 16);
        q1 = _mm256_loadu_ps(query + processed + 24);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        detail::expand_u8x16_to_ps(vec_32_to_47, vals_lo, vals_hi);
        q0 = _mm256_loadu_ps(query + processed + 32);
        q1 = _mm256_loadu_ps(query + processed + 40);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        detail::expand_u8x16_to_ps(vec_48_to_63, vals_lo, vals_hi);
        q0 = _mm256_loadu_ps(query + processed + 48);
        q1 = _mm256_loadu_ps(query + processed + 56);
        total += detail::hsum_ps(_mm256_add_ps(_mm256_mul_ps(q0, vals_lo), _mm256_mul_ps(q1, vals_hi)));

        processed += 64;
    }

    for (; processed < dim; ++processed) {
        size_t block = processed / 64;
        size_t offset = processed % 64;
        const uint8_t* base_ptr = compact_code + block * 56;
        const uint8_t* cpt1 = base_ptr;
        const uint8_t* cpt2 = base_ptr + 16;
        const uint8_t* cpt3 = base_ptr + 32;
        int64_t top_bits = *reinterpret_cast<const int64_t*>(base_ptr + 48);

        uint8_t low;
        if (offset < 16) {
            low = cpt1[offset] & 0x3F;
        } else if (offset < 32) {
            low = cpt2[offset - 16] & 0x3F;
        } else if (offset < 48) {
            low = cpt3[offset - 32] & 0x3F;
        } else {
            size_t idx = offset - 48;
            uint8_t part1 = static_cast<uint8_t>((cpt1[idx] & 0xC0) >> 6);
            uint8_t part2 = static_cast<uint8_t>((cpt2[idx] & 0xC0) >> 4);
            uint8_t part3 = static_cast<uint8_t>((cpt3[idx] & 0xC0) >> 2);
            low = static_cast<uint8_t>(part1 | part2 | part3);
        }

        uint8_t top = static_cast<uint8_t>(detail::extract_interleaved_bit(top_bits, offset));
        uint8_t value = static_cast<uint8_t>(low | (top << 6));
        total += query[processed] * static_cast<float>(value);
    }

    return total;
#else
    float result = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        size_t block = i / 64;
        size_t offset = i % 64;
        const uint8_t* base_ptr = compact_code + block * 56;
        const uint8_t* cpt1 = base_ptr;
        const uint8_t* cpt2 = base_ptr + 16;
        const uint8_t* cpt3 = base_ptr + 32;
        int64_t top_bits = *reinterpret_cast<const int64_t*>(base_ptr + 48);

        uint8_t low;
        if (offset < 16) {
            low = cpt1[offset] & 0x3F;
        } else if (offset < 32) {
            low = cpt2[offset - 16] & 0x3F;
        } else if (offset < 48) {
            low = cpt3[offset - 32] & 0x3F;
        } else {
            size_t idx = offset - 48;
            uint8_t part1 = static_cast<uint8_t>((cpt1[idx] & 0xC0) >> 6);
            uint8_t part2 = static_cast<uint8_t>((cpt2[idx] & 0xC0) >> 4);
            uint8_t part3 = static_cast<uint8_t>((cpt3[idx] & 0xC0) >> 2);
            low = static_cast<uint8_t>(part1 | part2 | part3);
        }

        uint8_t top = static_cast<uint8_t>(detail::extract_interleaved_bit(top_bits, offset));
        uint8_t value = static_cast<uint8_t>(low | (top << 6));
        result += query[i] * static_cast<float>(value);
    }
    return result;
#endif
}

// inner product between float type and int type vectors
template <typename TF, typename TI>
inline TF ip_fxi(const TF* __restrict__ vec0, const TI* __restrict__ vec1, size_t dim) {
    static_assert(std::is_floating_point_v<TF>, "TF must be an floating type");
    static_assert(std::is_integral_v<TI>, "TI must be an integeral type");

    ConstVectorMap<TF> v0(vec0, dim);
    ConstVectorMap<TI> v1(vec1, dim);
    return v0.dot(v1.template cast<TF>());
}
}  // namespace excode_ipimpl

using ex_ipfunc = float (*)(const float*, const uint8_t*, size_t);

inline ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 1) {
        // when ex_bits = 0, we do not use it
        return excode_ipimpl::ip16_fxu1_avx2;
    }
    if (ex_bits == 2) {
        return excode_ipimpl::ip16_fxu2_avx2;
    }
    if (ex_bits == 3) {
        return excode_ipimpl::ip64_fxu3_avx2;
    }
    if (ex_bits == 4) {
        return excode_ipimpl::ip16_fxu4_avx2;
    }
    if (ex_bits == 5) {
        return excode_ipimpl::ip64_fxu5_avx2;
    }
    if (ex_bits == 6) {
        return excode_ipimpl::ip16_fxu6_avx2;
    }
    if (ex_bits == 7) {
        return excode_ipimpl::ip64_fxu7_avx2;
    }
    if (ex_bits == 8) {
        return excode_ipimpl::ip_fxi;
    }

    std::cerr << "Bad IP function for IVF\n";
    exit(1);
}

static inline uint32_t reverse_bits(uint32_t n) {
    n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
    n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
    n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
    n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
    return n;
}

static inline uint64_t reverse_bits_u64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555) | ((n << 1) & 0xaaaaaaaaaaaaaaaa);
    n = ((n >> 2) & 0x3333333333333333) | ((n << 2) & 0xcccccccccccccccc);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0f) | ((n << 4) & 0xf0f0f0f0f0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff00ff00ff) | ((n << 8) & 0xff00ff00ff00ff00);
    n = ((n >> 16) & 0x0000ffff0000ffff) | ((n << 16) & 0xffff0000ffff0000);
    n = ((n >> 32) & 0x00000000ffffffff) | ((n << 32) & 0xffffffff00000000);
    return n;
}

// shuffle control: extract the low byte of each 16-bit lane into the
// lower 8 bytes of each 128-bit lane. Control bytes with high bit set
// zero the corresponding output byte.
const __m256i shuffle_ctl = _mm256_setr_epi8(
    0, 2, 4, 6, 8, 10, 12, 14, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    0, 2, 4, 6, 8, 10, 12, 14, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80
);

inline void transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
#if defined(__AVX512F__)
    // 512 / 16 = 32
    for (size_t i = 0; i < padded_dim; i += 32) {
        __m512i v = _mm512_loadu_si512(q);
        v = _mm512_slli_epi32(v, (16 - b_query));
        for (size_t j = 0; j < b_query; ++j) {
            uint32_t v1 = _mm512_movepi16_mask(v);  // get most significant bit
            v1 = reverse_bits(v1);
            tq[((b_query - j - 1) * (padded_dim / 64)) + (i / 64)] |=
                (static_cast<uint64_t>(v1) << ((i / 32 % 2 == 0) ? 32 : 0));
            v = _mm512_add_epi32(v, v);
        }
        q += 32;
    }
#elif defined(__AVX2__)
    const int shift_bits = (b_query >= 16) ? 0 : static_cast<int>(16 - b_query);
    const __m128i shift = _mm_cvtsi32_si128(shift_bits);
    for (size_t i = 0; i < padded_dim; i += 32) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16));

        v0 = _mm256_sll_epi16(v0, shift);
        v1 = _mm256_sll_epi16(v1, shift);

        for (size_t j = 0; j < b_query; ++j) {
            __m256i bit0 = _mm256_srai_epi16(v0, 15);
            __m256i bit1 = _mm256_srai_epi16(v1, 15);
            __m256i packed = _mm256_packs_epi16(bit0, bit1);

            uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(packed));
            mask = reverse_bits(mask);

            size_t idx = ((b_query - j - 1) * (padded_dim / 64)) + (i / 64);
            size_t shift_offset = ((i / 32) % 2 == 0) ? 32 : 0;
            tq[idx] |= static_cast<uint64_t>(mask) << shift_offset;

            v0 = _mm256_slli_epi16(v0, 1);
            v1 = _mm256_slli_epi16(v1, 1);
        }

        q += 32;
    }
#else
    const uint16_t shift_bits = (b_query >= 16) ? 0 : static_cast<uint16_t>(16 - b_query);
    for (size_t i = 0; i < padded_dim; i += 32) {
        uint16_t block[32];
        for (size_t lane = 0; lane < 32; ++lane) {
            block[lane] = static_cast<uint16_t>(q[lane] << shift_bits);
        }

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t mask = 0;
            for (size_t lane = 0; lane < 32; ++lane) {
                mask |= static_cast<uint32_t>((block[lane] >> 15) & 0x1U) << lane;
                block[lane] = static_cast<uint16_t>(block[lane] << 1);
            }

            mask = reverse_bits(mask);
            size_t idx = ((b_query - j - 1) * (padded_dim / 64)) + (i / 64);
            size_t shift_offset = ((i / 32) % 2 == 0) ? 32 : 0;
            tq[idx] |= static_cast<uint64_t>(mask) << shift_offset;
        }

        q += 32;
    }
#endif
}

// Emulate _mm512_movepi16_mask for a 64-element uint16_t array on AVX2
// Returns a 64-bit mask where bit i is the most-significant-bit of data[i]
inline uint64_t movepi16_mask_64(const uint16_t* data) {
#if defined(__AVX2__)

    uint64_t result = 0;

    // Process 4 blocks of 16 uint16_t (16*4 = 64)
    for (int block = 0; block < 4; ++block) {
        const uint16_t* base = data + block * 16;
        // load 16 uint16_t -> 256 bits
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));

        // todo: 16 << 12 ???
        v = _mm256_slli_epi32(v, 12);

        // arithmetic shift right by 15 to broadcast the sign bit: 0x0000 or 0xFFFF
        __m256i signs = _mm256_srai_epi16(v, 15);

        // shuffle to extract the low byte of each 16-bit lane into positions 0..7
        __m256i bytes = _mm256_shuffle_epi8(signs, shuffle_ctl);

        // extract low and high 128-bit lanes
        __m128i low128 = _mm256_castsi256_si128(bytes);
        __m128i high128 = _mm256_extracti128_si256(bytes, 1);

        // movemask on 128 yields 16-bit mask; only lower 8 bits contain our data
        uint32_t mlow = static_cast<uint32_t>(_mm_movemask_epi8(low128)) & 0xFFu;
        uint32_t mhigh = static_cast<uint32_t>(_mm_movemask_epi8(high128)) & 0xFFu;

        uint32_t block_mask = mlow | (mhigh << 8);

        result |= (uint64_t)block_mask << (block * 16);
    }

    return result;
#else
    // fallback scalar
    uint64_t r = 0;
    for (int i = 0; i < 64; ++i) {
        r |= (uint64_t)((data[i] & 0x8000u) ? 1u : 0u) << i;
    }
    return r;
#endif
}

// Emulate _mm512_movepi16_mask for a 64-element uint16_t array on AVX2
// Returns a 64-bit mask where bit i is the most-significant-bit of data[i]
inline uint16_t movepi16_mask_64(__m256i v) {
    // uint64_t result = 0;
    // int block = 0; //to 4
    // const uint16_t* base = data + block * 16;
    // load 16 uint16_t -> 256 bits
    // __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));
    //
    // // : 16 << 12 ???
    // v = _mm256_slli_epi32(v, 12);

    // arithmetic shift right by 15 to broadcast the sign bit: 0x0000 or 0xFFFF
    __m256i signs = _mm256_srai_epi16(v, 15);

    // shuffle to extract the low byte of each 16-bit lane into positions 0..7
    __m256i bytes = _mm256_shuffle_epi8(signs, shuffle_ctl);

    // extract low and high 128-bit lanes
    __m128i low128 = _mm256_castsi256_si128(bytes);
    __m128i high128 = _mm256_extracti128_si256(bytes, 1);

    // movemask on 128 yields 16-bit mask; only lower 8 bits contain our data
    uint32_t mlow = static_cast<uint32_t>(_mm_movemask_epi8(low128)) & 0xFFu;
    uint32_t mhigh = static_cast<uint32_t>(_mm_movemask_epi8(high128)) & 0xFFu;

    uint32_t block_mask = mlow | (mhigh << 8);

    // result |= (uint64_t)block_mask << (block * 16);

    return block_mask;
}





static inline void new_transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {

#if defined(__AVX512F__)

    // 512 / 16 = 32
    const uint16_t* q_cpy = q;
    const uint64_t* tq_cpy = tq;

    std::vector<uint64_t> tq_verifyer;
    // padded_dim = 768 ( / 64 = 12)
    // b_query = 4
    tq_verifyer.resize(48);

    for (size_t i = 0; i < padded_dim; i += 64) {
        __m512i vec_00_to_31 = _mm512_loadu_si512(q);
        __m512i vec_32_to_63 = _mm512_loadu_si512(q + 32);

        // the first (16 - b_query) bits are empty
        vec_00_to_31 = _mm512_slli_epi32(vec_00_to_31, (16 - b_query));
        vec_32_to_63 = _mm512_slli_epi32(vec_32_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t v0 = _mm512_movepi16_mask(vec_00_to_31);  // get most significant bit
            uint32_t v1 = _mm512_movepi16_mask(vec_32_to_63);  // get most significant bit


            // [TODO: remove all reverse_bits]
            v0 = reverse_bits(v0);
            v1 = reverse_bits(v1);
            uint64_t v = (static_cast<uint64_t>(v0) << 32) + v1;

            tq[b_query - j - 1] = v;
            tq_verifyer[b_query - j - 1 + i / 64 * b_query] = v;

            vec_00_to_31 = _mm512_slli_epi16(vec_00_to_31, 1);
            vec_32_to_63 = _mm512_slli_epi16(vec_32_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
#elif defined(__AVX2__)
    uint32_t shift = 16 - b_query;
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m256i vec_00_to_15 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        __m256i vec_16_to_31 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16));
        __m256i vec_32_to_47 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
        __m256i vec_48_to_63 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 48));


        vec_00_to_15 = _mm256_slli_epi16(vec_00_to_15, shift);
        vec_16_to_31 = _mm256_slli_epi16(vec_16_to_31, shift);
        vec_32_to_47 = _mm256_slli_epi16(vec_32_to_47, shift);
        vec_48_to_63 = _mm256_slli_epi16(vec_48_to_63, shift);

        for (size_t j = 0; j < b_query; ++j) {
            uint64_t test0 = movepi16_mask_64(vec_00_to_15);
            uint64_t test1 = movepi16_mask_64(vec_16_to_31);
            uint64_t test2 = movepi16_mask_64(vec_32_to_47);
            uint64_t test3 = movepi16_mask_64(vec_48_to_63);

            // test0 = (reverse_bits(test0) << 16) | reverse_bits(test1);
            // test2 = (reverse_bits(test2)) | reverse_bits(test3);

            test0 = test1 << 16 | test0;
            test2 = test3 << 16 | test2;


            test0 = reverse_bits(test0);
            test2 = reverse_bits(test2);


            uint64_t v = (static_cast<uint64_t>(test0) << 32) | test2;
            tq[b_query - j - 1] = v;

            vec_00_to_15 = _mm256_slli_epi16(vec_00_to_15, 1);
            vec_16_to_31 = _mm256_slli_epi16(vec_16_to_31, 1);
            vec_32_to_47 = _mm256_slli_epi16(vec_32_to_47, 1);
            vec_48_to_63 = _mm256_slli_epi16(vec_48_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
#else
    const uint16_t shift_bits = (b_query >= 16) ? 0 : static_cast<uint16_t>(16 - b_query);
    for (size_t i = 0; i < padded_dim; i += 64) {
        uint16_t block[64];
        for (size_t lane = 0; lane < 64; ++lane) {
            block[lane] = static_cast<uint16_t>(q[lane] << shift_bits);
        }

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t mask0 = 0;
            uint32_t mask1 = 0;
            for (size_t lane = 0; lane < 32; ++lane) {
                mask0 |= static_cast<uint32_t>((block[lane] >> 15) & 0x1U) << lane;
                mask1 |= static_cast<uint32_t>((block[lane + 32] >> 15) & 0x1U) << lane;
                block[lane] = static_cast<uint16_t>(block[lane] << 1);
                block[lane + 32] = static_cast<uint16_t>(block[lane + 32] << 1);
            }

            mask0 = reverse_bits(mask0);
            mask1 = reverse_bits(mask1);
            tq[b_query - j - 1] = (static_cast<uint64_t>(mask0) << 32) | mask1;
        }

        tq += b_query;
        q += 64;
    }
#endif
}

inline float mask_ip_x0_q_old(const float* query, const uint64_t* data, size_t padded_dim) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        __mmask16 mask0 = static_cast<__mmask16>(bits >> 00);  // for q[0..15]
        __mmask16 mask1 = static_cast<__mmask16>(bits >> 16);  // for q[16..31]
        __mmask16 mask2 = static_cast<__mmask16>(bits >> 32);  // for q[32..47]
        __mmask16 mask3 = static_cast<__mmask16>(bits >> 48);  // for q[48..63]

        __m512 q0 = _mm512_loadu_ps(it_query);
        __m512 q1 = _mm512_loadu_ps(it_query + 16);
        __m512 q2 = _mm512_loadu_ps(it_query + 32);
        __m512 q3 = _mm512_loadu_ps(it_query + 48);

        __m512 masked0 = _mm512_maskz_mov_ps(mask0, q0);
        __m512 masked1 = _mm512_maskz_mov_ps(mask1, q1);
        __m512 masked2 = _mm512_maskz_mov_ps(mask2, q2);
        __m512 masked3 = _mm512_maskz_mov_ps(mask3, q3);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        it_data++;
        it_query += 64;
    }
    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    float total = 0.0F;
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        for (size_t block = 0; block < 4; ++block) {
            uint16_t mask16 = static_cast<uint16_t>(bits >> (block * 16));
            uint8_t mask_lo = static_cast<uint8_t>(mask16 & 0xFF);
            uint8_t mask_hi = static_cast<uint8_t>(mask16 >> 8);

            const float* block_ptr = it_query + block * 16;

            __m256 q_lo = _mm256_loadu_ps(block_ptr);
            __m256 q_hi = _mm256_loadu_ps(block_ptr + 8);

            __m256 mask_vec_lo = _mm256_castsi256_ps(detail::mask_from_u8(mask_lo));
            __m256 mask_vec_hi = _mm256_castsi256_ps(detail::mask_from_u8(mask_hi));

            total += detail::hsum_ps(_mm256_and_ps(q_lo, mask_vec_lo));
            total += detail::hsum_ps(_mm256_and_ps(q_hi, mask_vec_hi));
        }

        it_data++;
        it_query += 64;
    }
    return total;
#else
    float total = 0.0F;
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        for (size_t j = 0; j < 64; ++j) {
            if ((bits >> j) & 0x1ULL) {
                total += it_query[j];
            }
        }
        it_data++;
        it_query += 64;
    }
    return total;
#endif
}

inline float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;

#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

        __mmask16 mask0 = static_cast<__mmask16>(bits);
        __mmask16 mask1 = static_cast<__mmask16>(bits >> 16);
        __mmask16 mask2 = static_cast<__mmask16>(bits >> 32);
        __mmask16 mask3 = static_cast<__mmask16>(bits >> 48);

        __m512 masked0 = _mm512_maskz_loadu_ps(mask0, it_query);
        __m512 masked1 = _mm512_maskz_loadu_ps(mask1, it_query + 16);
        __m512 masked2 = _mm512_maskz_loadu_ps(mask2, it_query + 32);
        __m512 masked3 = _mm512_maskz_loadu_ps(mask3, it_query + 48);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        //         _mm_prefetch(reinterpret_cast<const char*>(it_query + 128), _MM_HINT_T1);

        ++it_data;
        it_query += 64;
    }

    //    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    float total = 0.0F;
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        for (size_t block = 0; block < 4; ++block) {
            uint16_t mask16 = static_cast<uint16_t>(bits >> (block * 16));
            uint8_t mask_lo = static_cast<uint8_t>(mask16 & 0xFF);
            uint8_t mask_hi = static_cast<uint8_t>(mask16 >> 8);

            const float* block_ptr = it_query + block * 16;

            __m256 q_lo = _mm256_loadu_ps(block_ptr);
            __m256 q_hi = _mm256_loadu_ps(block_ptr + 8);

            __m256 mask_vec_lo = _mm256_castsi256_ps(detail::mask_from_u8(mask_lo));
            __m256 mask_vec_hi = _mm256_castsi256_ps(detail::mask_from_u8(mask_hi));

            total += detail::hsum_ps(_mm256_and_ps(q_lo, mask_vec_lo));
            total += detail::hsum_ps(_mm256_and_ps(q_hi, mask_vec_hi));
        }

        ++it_data;
        it_query += 64;
    }

    return total;
#else
    float total = 0.0F;
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        for (size_t j = 0; j < 64; ++j) {
            if ((bits >> j) & 0x1ULL) {
                total += it_query[j];
            }
        }

        ++it_data;
        it_query += 64;
    }

    return total;
#endif
}

inline float ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

    size_t ip = 0;
    size_t ppc = 0;

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t x = *static_cast<const uint64_t*>(it_data);
        ppc += __builtin_popcountll(x);

        for (size_t j = 0; j < b_query; ++j) {
            uint64_t y = *static_cast<const uint64_t*>(it_query);
            ip += (__builtin_popcountll(x & y) << j);
            it_query++;
        }
        it_data++;
    }

    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}

static inline uint32_t ip_bin_bin(const uint64_t* q, const uint64_t* d, size_t padded_dim) {
    uint64_t ret = 0;
    size_t iter = padded_dim / 64;
    for (size_t i = 0; i < iter; ++i) {
        ret += __builtin_popcountll((*d) & (*q));
        q++;
        d++;
    }
    return ret;
}

inline uint32_t ip_byte_bin(
    const uint64_t* q, const uint64_t* d, size_t padded_dim, size_t b_query
) {
    uint32_t ret = 0;
    size_t offset = (padded_dim / 64);
    for (size_t i = 0; i < b_query; i++) {
        ret += (ip_bin_bin(q, d, padded_dim) << i);
        q += offset;
    }
    return ret;
}

inline size_t popcount(const uint64_t* __restrict__ d, size_t length) {
    size_t ret = 0;
    for (size_t i = 0; i < length / 64; ++i) {
        ret += __builtin_popcountll((*d));
        ++d;
    }
    return ret;
}

template <typename T>
RowMajorMatrix<T> random_gaussian_matrix(size_t rows, size_t cols) {
    RowMajorMatrix<T> rand(rows, cols);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, 1);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            rand(i, j) = dist(gen);
        }
    }

    return rand;
}
}  // namespace rabitqlib
