#pragma once

#include "core.h"
#include "util.h"
#include "../../third_party/hnswlib/hnswlib.h"
#include <cstddef>
#include <utility>

class SolveStrategy {
public:
    SolveStrategy(std::string source_path, std::string query_path, std::string index_path) {
        // Read data and query
        ReadData(source_path, data_set_, data_num_, data_dim_);
        ReadData(query_path, query_set_, query_num_, query_dim_);
        knn_results_.resize(query_num_);
        M_ = M;
        M0_ = M0;
        ef_construction_ = EF_CONSTRUCTION;
        ef_search_ = EF_SEARCH;
        K_ = K;
        branching_factor_ = BRANCHING_FACTOR;
        threshold_level_ = THRESHOLD_LEVEL;

        index_path_ = index_path;
    }

    SolveStrategy(std::string source_path, std::string query_path, std::string index_path,
                  int partial, size_t&max_size) {
        // Read data and query
        ReadData(source_path, data_set_, data_num_, data_dim_, partial, max_size);
        ReadData(query_path, query_set_, query_num_, query_dim_);
        knn_results_.resize(query_num_);
        M_ = M;
        M0_ = M0;
        ef_construction_ = EF_CONSTRUCTION;
        ef_search_ = EF_SEARCH;
        K_ = K;
        branching_factor_ = BRANCHING_FACTOR;
        threshold_level_ = THRESHOLD_LEVEL;

        index_path_ = index_path;
    }

    virtual ~SolveStrategy() = default;

    virtual void solve() = 0;

    void setEf(size_t ef) {
        ef_search_ = ef;
    }

    void clear() {
        data_set_.clear();
        query_set_.clear();
        knn_results_.clear();
    }

    void read_knn(std::string knn_path) {
        uint32_t num, dim;
        ReadData(knn_path, knn_results_, num, dim);
    }

    void save_knn(std::string knn_path) {
        WriteData(knn_path, knn_results_);
    }

    void recall(std::string gt_path) {
        // Read ground truth
        uint32_t gt_num, gt_dim;
        std::vector<std::vector<uint32_t>> gt_set;
        ReadData(gt_path, gt_set, gt_num, gt_dim);

        // Calculate recall
        std::atomic<int> hit = 0;
        size_t dim = data_dim_;
#pragma omp parallel for schedule(dynamic)
        for (uint32_t i = 0; i < query_num_; ++i) {
            auto& knn = knn_results_[i];
            // auto& truth_knn = gt_set[i];
            std::vector<uint32_t> truth_knn;

            // fetch the top-K ground truth
            std::vector<std::pair<float, uint32_t>> knn_with_dist;
            for (auto gt : gt_set[i]) {
                knn_with_dist.emplace_back(std::make_pair(hnswlib::L2Sqr(query_set_[i].data(), data_set_[gt].data(), &dim), gt));
            }
            sort(knn_with_dist.begin(), knn_with_dist.end());
            truth_knn.clear();
            for (size_t j = 0; j < K; ++j) {
                truth_knn.emplace_back(knn_with_dist[j].second);
            }

            std::sort(knn.begin(), knn.end());
            std::sort(truth_knn.begin(), truth_knn.end());

            std::vector<uint32_t> intersection;
            std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(), truth_knn.end(), std::back_inserter(intersection));
            hit.fetch_add(intersection.size());
        }

        float recall = static_cast<float>(hit.load()) / (query_num_ * K);
        std::cout << "Recall: " << recall << std::endl;
    }
 
  protected:
    // data
    std::vector<std::vector<float>> data_set_;
    uint32_t data_num_;
    uint32_t data_dim_;
    size_t M_;
    size_t M0_;
    size_t ef_construction_;
    std::string branching_factor_;
    size_t threshold_level_;

    // query
    std::vector<std::vector<float>> query_set_;
    uint32_t query_num_;
    uint32_t query_dim_;
    size_t ef_search_;

    // knn_results
    std::vector<std::vector<uint32_t>> knn_results_;
    size_t K_;

    std::string index_path_;
};
