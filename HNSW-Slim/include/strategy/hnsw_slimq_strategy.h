#pragma once

#include "../../third_party/hnswlib/hnswalg_slimq.h"

#include "core.h"
#include "hnswlib/hnswalg.h"
#include "solve_strategy.h"
#include <csignal>

#include <iostream>
#include <string>

#include "../../third_party/rabitqlib/index/hnsw/hnsw.hpp"
#include "../../third_party/rabitqlib/defines.hpp"
#include "../../third_party/rabitqlib/utils/io.hpp"
#include "../../third_party/rabitqlib/utils/stopw.hpp"

using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

class HnswSlimQStrategy : public SolveStrategy {
public:
  HnswSlimQStrategy(std::string source_path, std::string query_path,
                 std::string index_path,
                     int threshold_level = 0,
                     float top_degree_percent0 = 0.02f,
                     float top_degree_percent = 0.02f,
                     size_t top_degree_M0 = 32,
                     size_t low_degree_m0 = 8,
                     size_t top_degree_M = 16,
                     size_t low_degree_m = 4,
                     int partial = 100,
                     std::string index_suffix = ""
                )
      : SolveStrategy(source_path, query_path, index_path, partial, max_size_),
        threshold_level_(threshold_level),
        top_degree_percent0_(top_degree_percent0), top_degree_percent_(top_degree_percent),
        top_degree_M0_(top_degree_M0), low_degree_m0_(low_degree_m0),
        top_degree_M_(top_degree_M), low_degree_m_(low_degree_m), partial_(partial),
        index_suffix_(index_suffix),
        source_path_(source_path) {
    centroid_path_ = source_path;
    cluster_path_ = source_path;
    centroid_path_.replace(centroid_path_.find("_base.fvecs"), 11, "_centroids_16.fvecs");
    cluster_path_.replace(cluster_path_.find("_base.fvecs"), 11, "_clusterids_16.ivecs");
  }

  void solve() {
    std::cout << "index path: " << index_path_ << std::endl;

    // Build HNSW index
    hnswlib::L2Space l2space(data_dim_);
    hnswlib::HierarchicalNSWSlimQ<float> hnsw_slimq(&l2space, data_num_, M_,
                                               ef_construction_, threshold_level_,
                                               top_degree_percent0_, top_degree_percent_,
                                               top_degree_M0_, low_degree_m0_,
                                               top_degree_M_, low_degree_m_);
    std::string hnsw_index_path = index_path_;
    hnsw_index_path.replace(hnsw_index_path.find("hnsw_slimq"), 10, "hnsw");

    if (index_suffix_.size() > 0) {
      hnsw_index_path.replace(hnsw_index_path.find(index_suffix_),
              index_suffix_.size(), "");
    }

    std::cout << "solve with index_path " << hnsw_index_path << std::endl;
    std::cout << "solve with index_path " << index_path_ << std::endl;

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    bool faster_quant = true;

    if (std::filesystem::exists(index_path_)) {
      hnsw_slimq.loadIndex(index_path_, &l2space, data_num_);
    } else {
      std::cout << "max_size: " << max_size_ << std::endl;
      if (std::filesystem::exists(hnsw_index_path)) {
        data_type data;
        rabitqlib::load_vecs<float, data_type>(source_path_.c_str(), data);
        rabitqlib::hnsw::HierarchicalNSW hnsw;
        hnsw.load(hnsw_index_path.c_str(), metric_type);
        hnsw.setRawData(data.data());

        auto s_build = std::chrono::system_clock::now();
        hnsw_slimq.convertFromHNSW(&hnsw);
        auto e_build = std::chrono::system_clock::now();
        std::cout << "convert hnsw to hnsw_slimq cost: "
                  << time_cost(s_build, e_build) << " (ms)\n";
        {
          std::filesystem::path fsPath(index_path_);
          fsPath.remove_filename();
          std::filesystem::create_directories(fsPath);
          std::ofstream out(index_path_, std::ios::binary);
          std::cout << "save index: " + index_path_ << std::endl;
          hnsw_slimq.saveIndex(index_path_);
        }
      } else {
        data_type centroids;
        rabitqlib::load_vecs<float, data_type>(centroid_path_.c_str(), centroids);
        data_type data;
        rabitqlib::load_vecs<float, data_type>(source_path_.c_str(), data);
        gt_type cluster_id;
        rabitqlib::load_vecs<uint32_t, gt_type>(cluster_path_.c_str(), cluster_id);

        size_t num_points = data_num_;
        size_t dim = data_dim_;
        auto* hnsw = new rabitqlib::hnsw::HierarchicalNSW(
            num_points, dim, 4, 32, 128, 100, metric_type
        );

        rabitqlib::StopW stopw;
        stopw.reset();

        auto hnsw_s_build = std::chrono::system_clock::now();
        hnsw->construct(
            centroids.rows(),
            centroids.data(),
            num_points,
            data.data(),
            cluster_id.data(),
            0,
            faster_quant
        );
        auto hnsw_e_build = std::chrono::system_clock::now();
        std::cout << "convert hnsw to hnsw_slimq cost: "
                  << time_cost(hnsw_s_build, hnsw_e_build) << " (ms)\n";

        hnsw->save(hnsw_index_path.c_str());
        auto s_build = std::chrono::system_clock::now();
        hnsw_slimq.convertFromHNSW(hnsw);
        auto e_build = std::chrono::system_clock::now();
        std::cout << "convert hnsw to hnsw_slimq cost: "
                  << time_cost(s_build, e_build) << " (ms)\n";
        {
          std::filesystem::path fsPath(index_path_);
          fsPath.remove_filename();
          std::filesystem::create_directories(fsPath);
          std::ofstream out(index_path_, std::ios::binary);
          std::cout << "save index: " + index_path_ << std::endl;
          hnsw_slimq.saveIndex(index_path_);
        }
      }
    }
    std::cout << "hnsw_slimq index size: " << hnsw_slimq.indexSize() << " bytes\n";

    hnsw_slimq.setDataset(&data_set_);
    if (partial_ == 100) {
      // Solve query
      hnsw_slimq.setEf(ef_search_);
      for (uint32_t i = 0; i < query_num_; ++i) {
        knn_results_[i].resize(K);
      }
      auto s_solve = std::chrono::system_clock::now();
      double query_time = 0;
      auto s_query = std::chrono::system_clock::now();

      // #pragma omp parallel for schedule(dynamic)
      for (uint32_t i = 0; i < query_num_; ++i) {
        hnsw_slimq.searchKnn(query_set_[i].data(), K, knn_results_[i].data());
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

  size_t max_size_;
  int partial_{100};
  std::string source_path_{""};
  std::string index_suffix_{""};
  std::string centroid_path_{""};
  std::string cluster_path_{""};

};
