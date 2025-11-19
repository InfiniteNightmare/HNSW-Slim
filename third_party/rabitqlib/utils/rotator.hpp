#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/fht_avx.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {

enum class RotatorType : uint8_t { MatrixRotator, FhtKacRotator };

// abstract rotator
template <typename T>
class Rotator {
   protected:
    size_t dim_;
    size_t padded_dim_;

   public:
    explicit Rotator() = default;
    explicit Rotator(size_t dim, size_t padded_dim) : dim_(dim), padded_dim_(padded_dim) {};
    virtual ~Rotator() = default;
    virtual void rotate(const T* src, T* dst) const = 0;
    virtual void load(std::ifstream&) = 0;
    virtual void save(std::ofstream&) const = 0;
    [[nodiscard]] size_t size() const { return this->padded_dim_; }
};

namespace rotator_impl {

// get padding requirement for different rotator
inline size_t padding_requirement(size_t dim, RotatorType type) {
    if (type == RotatorType::MatrixRotator) {
        return dim;
    }
    if (type == RotatorType::FhtKacRotator) {
        return round_up_to_multiple(dim, 64);
    }
    std::cerr << "Invalid rotator type in padding_requirement()\n" << std::flush;
    exit(1);
}

template <typename T = float>
class MatrixRotator : public Rotator<T> {
   private:
    RowMajorMatrix<T> rand_mat_;  // Rotation Maxtrix
   public:
    explicit MatrixRotator(size_t dim, size_t padded_dim)
        : Rotator<T>(dim, padded_dim), rand_mat_(dim, padded_dim) {
        RowMajorMatrix<T> rand = random_gaussian_matrix<T>(padded_dim, padded_dim);
        Eigen::HouseholderQR<RowMajorMatrix<T>> qr(rand);
        RowMajorMatrix<T> q_inv =
            qr.householderQ().transpose();  // inverse of orthogonal mat is its inverse

        // the random matrix only need the first dim rows, since we just pad zeros for
        // the vector to be rotated to padded dimension
        std::memcpy(&rand_mat_(0, 0), &q_inv(0, 0), sizeof(T) * dim * padded_dim);
    }
    MatrixRotator() = default;
    ~MatrixRotator() = default;

    MatrixRotator& operator=(const MatrixRotator& other) {
        this->dim_ = other.dim_;
        this->padded_dim_ = other.padded_dim_;
        this->rand_mat_ = other.rand_mat_;
        return *this;
    }

    void load(std::ifstream& input) override {
        input.read(
            reinterpret_cast<char*>(rand_mat_.data()),
            static_cast<long>(sizeof(float) * this->dim_ * this->padded_dim_)
        );
    }

    void save(std::ofstream& output) const override {
        output.write(
            reinterpret_cast<const char*>(rand_mat_.data()),
            (sizeof(float) * this->dim_ * this->padded_dim_)
        );
    }

    void rotate(const T* vec, T* rotated_vec) const override {
        ConstRowMajorMatrixMap<T> v(vec, 1, this->dim_);
        RowMajorMatrixMap<T> rv(rotated_vec, 1, this->padded_dim_);
        rv = v * this->rand_mat_;
    }
};

static inline void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    constexpr size_t kFloatsPerChunk = 64;  // Process 64 floats per iteration
/*
#if defined(__AVX512F__)
    static_assert(
        kFloatsPerChunk % 16 == 0,
        "floats_per_chunk must be divisible by AVX512 register width"
    );

    for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
        uint64_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        const __mmask16 mask0 = _cvtu32_mask16(static_cast<uint32_t>(mask_bits & 0xFFFF));
        const __mmask16 mask1 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 16) & 0xFFFF));
        const __mmask16 mask2 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 32) & 0xFFFF));
        const __mmask16 mask3 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 48) & 0xFFFF));

        const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

        __m512 vec0 = _mm512_loadu_ps(&data[i]);
        vec0 = _mm512_mask_xor_ps(vec0, mask0, vec0, sign_flip);
        _mm512_storeu_ps(&data[i], vec0);

        __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
        vec1 = _mm512_mask_xor_ps(vec1, mask1, vec1, sign_flip);
        _mm512_storeu_ps(&data[i + 16], vec1);

        __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
        vec2 = _mm512_mask_xor_ps(vec2, mask2, vec2, sign_flip);
        _mm512_storeu_ps(&data[i + 32], vec2);

        __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
        vec3 = _mm512_mask_xor_ps(vec3, mask3, vec3, sign_flip);
        _mm512_storeu_ps(&data[i + 48], vec3);
    }
#elif defined(__AVX2__)
    */
#if defined(__AVX2__)
    const __m256 sign_flip = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
        uint64_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        for (size_t block = 0; block < 4; ++block) {
            uint16_t mask16 = static_cast<uint16_t>(mask_bits >> (block * 16));
            uint8_t mask_lo = static_cast<uint8_t>(mask16 & 0xFF);
            uint8_t mask_hi = static_cast<uint8_t>(mask16 >> 8);

            float* base = data + i + block * 16;

            // If both AVX512 and AVX2 are available at compile time but runtime
            // supports both, perform a verification by running an AVX512-like
            // sequence via scalar fallback into a temporary and compare.
#if defined(__AVX512F__)
            alignas(64) float tmp[16];
            // Emulate AVX512 mask-based flip into tmp using scalar ops for verification
            for (int t = 0; t < 16; ++t) {
                uint8_t bit = ((block * 16) + t) / 8;
                uint8_t bitpos = ((block * 16) + t) % 8;
                bool should_flip = ((flip[(i / 8) + bit] >> bitpos) & 0x1U);
                tmp[t] = should_flip ? -base[t] : base[t];
            }
#endif

            __m256 vec_lo = _mm256_loadu_ps(base);
            __m256 vec_hi = _mm256_loadu_ps(base + 8);

            __m256 mask_lo_ps = _mm256_castsi256_ps(detail::mask_from_u8(mask_lo));
            __m256 mask_hi_ps = _mm256_castsi256_ps(detail::mask_from_u8(mask_hi));

            vec_lo = _mm256_xor_ps(vec_lo, _mm256_and_ps(mask_lo_ps, sign_flip));
            vec_hi = _mm256_xor_ps(vec_hi, _mm256_and_ps(mask_hi_ps, sign_flip));

            _mm256_storeu_ps(base, vec_lo);
            _mm256_storeu_ps(base + 8, vec_hi);

#if defined(__AVX512F__)
            // Compare tmp vs produced result
            bool ok = true;
            for (int t = 0; t < 8; ++t) {
                if (!float_eq(tmp[t], base[t])) { ok = false; break; }
            }
            for (int t = 0; t < 8; ++t) {
                if (!float_eq(tmp[t + 8], base[t + 8])) { ok = false; break; }
            }
            if (!ok) {
                std::cerr << "[verify] flip_sign: AVX512 scalar emulate vs AVX2 result mismatch\n";
            }
#endif
        }
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        if ((flip[byte_idx] >> bit_idx) & 0x1U) {
            data[i] = -data[i];
        }
    }
#endif
}

class FhtKacRotator : public Rotator<float> {
   private:
    std::vector<uint8_t> flip_;
    std::function<void(float*)> fht_float_ = helper_float_6;
    size_t trunc_dim_ = 0;
    float fac_ = 0;

    static constexpr size_t kByteLen = 8;

   public:
    explicit FhtKacRotator(size_t dim, size_t padded_dim)
        : Rotator<float>(dim, padded_dim), flip_(4 * padded_dim / kByteLen) {
        std::random_device rd;   // Seed
        std::mt19937 gen(rd());  // Mersenne Twister RNG

        // Uniform distribution in the range [0, 255]
        std::uniform_int_distribution<int> dist(0, 255);

        // Generate a single random uint8_t value
        for (auto& i : flip_) {
            i = static_cast<uint8_t>(dist(gen));
        }

        // TODO(lib): is it portable?
        size_t bottom_log_dim = floor_log2(dim);
        trunc_dim_ = 1 << bottom_log_dim;
        fac_ = 1.0F / std::sqrt(static_cast<float>(trunc_dim_));

        switch (bottom_log_dim) {
            case 6:
                this->fht_float_ = helper_float_6;
                break;
            case 7:
                this->fht_float_ = helper_float_7;
                break;
            case 8:
                this->fht_float_ = helper_float_8;
                break;
            case 9:
                this->fht_float_ = helper_float_9;
                break;
            case 10:
                this->fht_float_ = helper_float_10;
                break;
            case 11:
                this->fht_float_ = helper_float_11;
                break;
            default:
                // TODO(lib): should we do more?
                std::cerr << "dimension of vector is too big\n";
                exit(1);
        }
    }
    FhtKacRotator() = default;
    ~FhtKacRotator() override = default;

    void load(std::ifstream& input) override {
        input.read(
            reinterpret_cast<char*>(flip_.data()),
            static_cast<long>(sizeof(uint8_t) * flip_.size())
        );
    }

    void save(std::ofstream& output) const override {
        output.write(
            reinterpret_cast<const char*>(flip_.data()),
            static_cast<long>(sizeof(uint8_t) * flip_.size())
        );
    }

    FhtKacRotator& operator=(const FhtKacRotator& other) {
        this->dim_ = other.dim_;
        this->padded_dim_ = other.padded_dim_;
        this->flip_ = other.flip_;
        this->fht_float_ = other.fht_float_;
        this->trunc_dim_ = other.trunc_dim_;
        this->fac_ = other.fac_;
        return *this;
    }

    void copyFlip(FhtKacRotator* target) const {
        if (target == nullptr) {
            return;
        }
        target->dim_ = this->dim_;
        target->padded_dim_ = this->padded_dim_;
        target->flip_ = flip_;
        target->fht_float_ = fht_float_;
        target->trunc_dim_ = trunc_dim_;
        target->fac_ = fac_;
    }

    static void kacs_walk(float* data, size_t len) {
#if defined(__AVX512F__) && defined(__AVX2__)
        // Run AVX512-like and AVX2 implementations and compare results.
        for (size_t i = 0; i < len / 2; i += 16) {
            // AVX512 path emulation using 16-wide ops
            __m512 x = _mm512_loadu_ps(&data[i]);
            __m512 y = _mm512_loadu_ps(&data[i + (len / 2)]);

            __m512 new_x = _mm512_add_ps(x, y);
            __m512 new_y = _mm512_sub_ps(x, y);

            // store tmp copy
            alignas(64) float tmp_x[16];
            alignas(64) float tmp_y[16];
            _mm512_storeu_ps(tmp_x, new_x);
            _mm512_storeu_ps(tmp_y, new_y);

            // AVX2 path to destination
            for (size_t j = 0; j < 16; j += 8) {
                __m256 ax = _mm256_loadu_ps(&data[i + j]);
                __m256 ay = _mm256_loadu_ps(&data[i + j + (len / 2)]);
                __m256 a_new_x = _mm256_add_ps(ax, ay);
                __m256 a_new_y = _mm256_sub_ps(ax, ay);
                _mm256_storeu_ps(&data[i + j], a_new_x);
                _mm256_storeu_ps(&data[i + j + (len / 2)], a_new_y);

                // compare
                alignas(32) float out_x[8];
                alignas(32) float out_y[8];
                _mm256_storeu_ps(out_x, a_new_x);
                _mm256_storeu_ps(out_y, a_new_y);
                for (int t = 0; t < 8; ++t) {
                    if (!float_eq(tmp_x[j + t], out_x[t]) || !float_eq(tmp_y[j + t], out_y[t])) {
                        std::cerr << "[verify] kacs_walk: AVX512 vs AVX2 mismatch at i=" << (i + j + t) << "\n";
                        break;
                    }
                }
            }
        }
#elif defined(__AVX512F__)
        for (size_t i = 0; i < len / 2; i += 16) {
            __m512 x = _mm512_loadu_ps(&data[i]);
            __m512 y = _mm512_loadu_ps(&data[i + (len / 2)]);

            __m512 new_x = _mm512_add_ps(x, y);
            __m512 new_y = _mm512_sub_ps(x, y);

            _mm512_storeu_ps(&data[i], new_x);
            _mm512_storeu_ps(&data[i + (len / 2)], new_y);
        }
#elif defined(__AVX2__)
        for (size_t i = 0; i < len / 2; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 y = _mm256_loadu_ps(&data[i + (len / 2)]);

            __m256 new_x = _mm256_add_ps(x, y);
            __m256 new_y = _mm256_sub_ps(x, y);

            _mm256_storeu_ps(&data[i], new_x);
            _mm256_storeu_ps(&data[i + (len / 2)], new_y);
        }
#else
        for (size_t i = 0; i < len / 2; ++i) {
            float x = data[i];
            float y = data[i + (len / 2)];
            data[i] = x + y;
            data[i + (len / 2)] = x - y;
        }
#endif
    }

    void rotate(const float* data, float* rotated_vec) const override {
        std::memcpy(rotated_vec, data, sizeof(float) * dim_);
        std::fill(rotated_vec + dim_, rotated_vec + padded_dim_, 0);

        if (trunc_dim_ == padded_dim_) {
            flip_sign(flip_.data(), rotated_vec, padded_dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_sign(flip_.data() + (padded_dim_ / kByteLen), rotated_vec, padded_dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_sign(
                flip_.data() + (2 * padded_dim_ / kByteLen), rotated_vec, padded_dim_
            );
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_sign(
                flip_.data() + (3 * padded_dim_ / kByteLen), rotated_vec, padded_dim_
            );
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            return;
        }

        size_t start = padded_dim_ - trunc_dim_;

        flip_sign(flip_.data(), rotated_vec, padded_dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        flip_sign(flip_.data() + (padded_dim_ / kByteLen), rotated_vec, padded_dim_);
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        flip_sign(flip_.data() + (2 * padded_dim_ / kByteLen), rotated_vec, padded_dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        flip_sign(flip_.data() + (3 * padded_dim_ / kByteLen), rotated_vec, padded_dim_);
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        // This can be removed if we don't care about the absolute value of
        // similarities.
        vec_rescale(rotated_vec, padded_dim_, 0.25F);
    }
};
}  // namespace rotator_impl

// for given dim & type, set rotator, return padded dimension
template <typename T>
Rotator<T>* choose_rotator(
    size_t dim, RotatorType type = RotatorType::FhtKacRotator, size_t padded_dim = 0
) {
    if (padded_dim == 0) {
        padded_dim = rotator_impl::padding_requirement(dim, type);
        if (padded_dim != dim) {
            std::cerr << "vectors are padded to " << padded_dim
                      << " dimensions for aligned computation\n";
            std::cerr << "check rabitqlib/utils/rotator.hpp in case that users want to "
                         "remove padding\n";
        }
    }

    if (padded_dim != rotator_impl::padding_requirement(padded_dim, type)) {
        std::cerr << "Invalid padded dim for the given rotator type\n" << std::flush;
        exit(1);
    }

    if (type == RotatorType::FhtKacRotator) {
        if (!std::is_same_v<T, float>) {
            std::cerr << "FhtKacRotator is only for float type currently\n";
            exit(1);
        }
        std::cerr << "FhtKacRotator is selected\n";
        return ::new rotator_impl::FhtKacRotator(dim, padded_dim);
    }

    if (type == RotatorType::MatrixRotator) {
        std::cerr << "MatrixRotator is selected\n";
        return ::new rotator_impl::MatrixRotator<T>(dim, padded_dim);
    }

    std::cerr << "Invaid rotator type in choose_rotator()\n";
    exit(1);
}
}  // namespace rabitqlib