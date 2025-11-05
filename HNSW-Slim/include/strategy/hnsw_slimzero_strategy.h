#pragma once

#include "../../third_party/hnswlib/hnswalg_slimzero.h"

#include "core.h"
#include "hnswlib/hnswalg.h"
#include "solve_strategy.h"
#include <csignal>

#include <iostream>
#include <string>

class HnswSlimZeroStrategy : public SolveStrategy {
public:
  HnswSlimZeroStrategy(std::string source_path, std::string query_path,
                      std::string index_path,
                     int threshold_level = 0,
                     float top_degree_percent0 = 0.02f,
                     float top_degree_percent = 0.02f,
                     size_t top_degree_M0 = 32,
                     size_t low_degree_m0 = 8,
                     size_t top_degree_M = 16,
                     size_t low_degree_m = 4,
                     size_t min_indegree0 = 8,
                     size_t min_indegree = 4,
                     std::string index_suffix = ""
                )
      : SolveStrategy(source_path, query_path, index_path),
        threshold_level_(threshold_level),
        top_degree_percent0_(top_degree_percent0), top_degree_percent_(top_degree_percent),
        top_degree_M0_(top_degree_M0), low_degree_m0_(low_degree_m0),
        top_degree_M_(top_degree_M), low_degree_m_(low_degree_m),
        min_indegree0_(min_indegree0), min_indegree_(min_indegree),
        index_suffix_(index_suffix) {
    max_size_ = data_num_;
  }

  void solve() {
    std::cout << "index path: " << index_path_ << std::endl;

    // Build HNSW index
    hnswlib::L2Space l2space(data_dim_);
    hnswlib::HierarchicalNSWSlimZero<float> hnsw_slim_zero(&l2space, data_num_, M_,
                                               ef_construction_, threshold_level_,
                                               top_degree_percent0_, top_degree_percent_,
                                               top_degree_M0_, low_degree_m0_,
                                               top_degree_M_, low_degree_m_,
                                               min_indegree0_, min_indegree_);
    std::string hnsw_index_path = index_path_;
    hnsw_index_path.replace(hnsw_index_path.find("hnsw_slimzero"), 13, "hnsw");

    if (index_suffix_.size() > 0) {
      hnsw_index_path.replace(hnsw_index_path.find(index_suffix_),
              index_suffix_.size(), "");
    }

    std::cout << "solve with index_path " << hnsw_index_path << std::endl;
    std::cout << "solve with index_path " << index_path_ << std::endl;


 if (std::filesystem::exists(index_path_)) {
      hnsw_slim_zero.loadIndex(index_path_, &l2space, data_num_);
    } else {
      std::cout << "max_size: " << max_size_ << std::endl;
      hnswlib::HierarchicalNSW<float> hnsw(&l2space, max_size_, M_,
                                           ef_construction_, branching_factor_);
      if (std::filesystem::exists(hnsw_index_path)) {
        hnsw.loadIndex(hnsw_index_path, &l2space, max_size_);
      } else {
        auto s_build = std::chrono::system_clock::now();
#pragma omp parallel for schedule(dynamic)
        for (uint32_t i = 0; i < data_num_; ++i) {
          hnsw.addPoint(data_set_[i].data(), i);
        }
        auto e_build = std::chrono::system_clock::now();
        std::cout << "build hnsw cost: " << time_cost(s_build, e_build)
                  << " (ms)\n";

        {
          std::filesystem::path fsPath(hnsw_index_path);
          fsPath.remove_filename();
          std::filesystem::create_directories(fsPath);
          std::ofstream out(hnsw_index_path, std::ios::binary);
          std::cout << "save hnsw index: " + hnsw_index_path << std::endl;
          hnsw.saveIndex(hnsw_index_path);
        }
      }
      std::cout << "hnsw index size: " << hnsw.indexSize() << " bytes\n";
      auto s_build = std::chrono::system_clock::now();
      hnsw_slim_zero.convertFromHNSW(&hnsw);
      auto e_build = std::chrono::system_clock::now();
      std::cout << "convert hnsw to hnsw_slim_zero cost: "
                << time_cost(s_build, e_build) << " (ms)\n";

      {
        std::filesystem::path fsPath(index_path_);
        fsPath.remove_filename();
        std::filesystem::create_directories(fsPath);
        std::ofstream out(index_path_, std::ios::binary);
        std::cout << "save index: " + index_path_ << std::endl;
        hnsw_slim_zero.saveIndex(index_path_);
      }
    }
    std::cout << "hnsw_slim_zero index size: " << hnsw_slim_zero.indexSize() << " bytes\n";

    float decay_fac = 1.0 / atoi(branching_factor_.c_str());

    float size_1 = 0, size_2 = 0, size_3 = 0;
    size_1 = 1.0 * data_num_ * 16 / 1024 / 1024;
    size_2 = 1.0 * 2 * data_num_ * decay_fac / (1 - decay_fac) / 1024 / 1024;

    if (threshold_level_ == 0) {
      size_3 = 4 * data_num_ * (2 + decay_fac) * (top_degree_percent_ * top_degree_M_
                    + (1 - top_degree_percent_) * low_degree_m_) / 1024 / 1024;
    } else {
      size_3 = 4 * data_num_ * (2 - decay_fac + std::pow(decay_fac,threshold_level_+1))
              * (top_degree_percent_ * top_degree_M_
                  + (1 - top_degree_percent_) * low_degree_m_) / 1024 / 1024;
    }
    std::cout << "estimation index size: " << size_1 + size_2 + size_3 << " bytes" << std::endl;

    if (partial_ == 100) {
      // Solve query
      hnsw_slim_zero.setEf(ef_search_);
      for (uint32_t i = 0; i < query_num_; ++i) {
        knn_results_[i].resize(K);
      }
      auto s_solve = std::chrono::system_clock::now();
      double query_time = 0;
      auto s_query = std::chrono::system_clock::now();

      // #pragma omp parallel for schedule(dynamic)
      for (uint32_t i = 0; i < query_num_; ++i) {
        hnsw_slim_zero.searchKnn(query_set_[i].data(), K, knn_results_[i].data());
      }
      auto e_solve = std::chrono::system_clock::now();
      auto e_query = std::chrono::system_clock::now();
      std::cout << "solve cost: " << time_cost(s_solve, e_solve) << " (ms)\n";
      std::cout << "query cost: " << std::chrono::duration<double>(e_query - s_query).count() << "\n";
    }
  }


private:
  int threshold_level_{0};
  float top_degree_percent0_{0.02f}; // \alpha_0\%
  float top_degree_percent_{0.02f};  // \alpha\%
  size_t top_degree_M0_{32};         // M_{h_0}
  size_t low_degree_m0_{8};          // M_{l_0}
  size_t top_degree_M_{16};          // M_{h}
  size_t low_degree_m_{4};           // M_{l}
  size_t min_indegree0_{8};          // min indegree for nodes in l0
  size_t min_indegree_{4};           // min indegree for nodes

  size_t max_size_;
  int partial_{100};
  std::string index_suffix_{""};
};
