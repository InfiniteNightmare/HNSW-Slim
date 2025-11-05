#pragma once

#include "../../third_party/hnswlib/bruteforce.h"
#include "solve_strategy.h"
#include <csignal>

class BruteForce : public SolveStrategy {
public:
  BruteForce(std::string source_path, std::string query_path,
             std::string index_path, std::string gt_path,
             size_t brute_k = 100)
      : SolveStrategy(source_path, query_path, index_path),
        gt_path_(gt_path), brute_k_(brute_k) {}

  void solve() {
    hnswlib::L2Space l2space(data_dim_);
    hnswlib::BruteforceSearch<float> bruteforce(&l2space, data_num_);

    for (uint32_t i = 0; i < data_num_; ++i) {
      bruteforce.addPoint(data_set_[i].data(), i);
    }

    knn_results_.resize(query_num_);
#pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < query_num_; ++i) {
      std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
          bruteforce.searchKnn(query_set_[i].data(), brute_k_);
      while (!result.empty() && knn_results_[i].size() < brute_k_) {
        knn_results_[i].emplace_back(result.top().second);
        result.pop();
      }
      while (knn_results_[i].size() < brute_k_) {
        knn_results_[i].emplace_back(-1);
        num_lack_++;
      }
    }

    std::cout << "gt_path: " << gt_path_ << std::endl;
    WriteData(gt_path_, knn_results_);

    std::cout << "num: " << knn_results_.size()
              << "dim: " << knn_results_[0].size() << std::endl;

    std::cout << "lack " << num_lack_ << std::endl;
  }

  private:
    std::string gt_path_;
    size_t brute_k_;
    size_t num_lack_{0};
};