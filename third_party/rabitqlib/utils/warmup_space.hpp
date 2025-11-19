#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

template <uint32_t b_query>
inline float warmup_ip_x0_q(
    const uint64_t* data,   // pointer to data blocks (each 64 bits)
    const uint64_t* query,  // pointer to query words (each 64 bits), arranged so that for
                            // each data block the corresponding b_query query words follow
    float delta,
    float vl,
    size_t padded_dim,
    [[maybe_unused]] size_t _b_query = 0  // not used
) {
    const size_t num_blk = padded_dim / 64;
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    constexpr size_t vec_width = 8;
    size_t vec_end = 0;

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    vec_end = (num_blk / vec_width) * vec_width;

    __m512i ip_vec = _mm512_setzero_si512();
    __m512i ppc_vec = _mm512_setzero_si512();

    for (size_t i = 0; i < vec_end; i += vec_width) {
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));

        __m512i popcnt_x_vec = _mm512_popcnt_epi64(x_vec);
        ppc_vec = _mm512_add_epi64(ppc_vec, popcnt_x_vec);

        __m512i block_ip = _mm512_setzero_si512();

        for (uint32_t j = 0; j < b_query; j++) {
            uint64_t indices[vec_width];
            for (size_t k = 0; k < vec_width; k++) {
                indices[k] = ((i + k) * b_query + j);
            }
            __m512i index_vec = _mm512_loadu_si512(indices);
            __m512i q_vec = _mm512_i64gather_epi64(index_vec, query, 8);

            __m512i and_vec = _mm512_and_si512(x_vec, q_vec);
            __m512i popcnt_and = _mm512_popcnt_epi64(and_vec);

            const uint64_t shift = 1ULL << j;
            __m512i shift_vec = _mm512_set1_epi64(shift);
            __m512i weighted = _mm512_mullo_epi64(popcnt_and, shift_vec);

            block_ip = _mm512_add_epi64(block_ip, weighted);
        }
        ip_vec = _mm512_add_epi64(ip_vec, block_ip);
    }

    uint64_t ip_arr[vec_width];
    uint64_t ppc_arr[vec_width];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(ip_arr), ip_vec);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(ppc_arr), ppc_vec);

    for (size_t k = 0; k < vec_width; k++) {
        ip_scalar += ip_arr[k];
        ppc_scalar += ppc_arr[k];
    }
#if defined(__AVX2__)
    // Optional verification for machines that support both AVX512VPOPCNTDQ and AVX2:
    // re-run the same vectorized loop with AVX2 scalarized popcnt and compare
    // aggregated results. This is relatively expensive so only performed when
    // compiling with both flags.
    {
        size_t ip_chk = 0;
        size_t ppc_chk = 0;
        for (size_t i = 0; i < vec_end; i += vec_width) {
            for (size_t k = 0; k < vec_width; ++k) {
                uint64_t x = data[i + k];
                ppc_chk += __builtin_popcountll(x);
                for (uint32_t j = 0; j < b_query; ++j) {
                    ip_chk += (__builtin_popcountll(x & query[(i + k) * b_query + j]) << j);
                }
            }
        }
        if (ip_chk != static_cast<size_t>(ip_scalar) || ppc_chk != static_cast<size_t>(ppc_scalar)) {
            std::cerr << "[verify] warmup_ip_x0_q: AVX512 popcnt vs AVX2 scalar mismatch\n";
        }
    }
#endif
#endif

    // Process remaining blocks that did not fit in the vectorized loop.
    for (size_t i = vec_end; i < num_blk; i++) {
        const uint64_t x = data[i];
        ppc_scalar += __builtin_popcountll(x);
        for (uint32_t j = 0; j < b_query; j++) {
            ip_scalar += __builtin_popcountll(x & query[i * b_query + j]) << j;
        }
    }

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}

template <uint32_t b_query, uint32_t padded_dim>
inline float warmup_ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t _padded_dim = 0,  // not used
    size_t _b_query = 0      // not used
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