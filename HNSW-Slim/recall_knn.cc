#include "core.h"
#include "strategy/solve_strategy.h"
#include "util.h"
#include <gflags/gflags.h>

DEFINE_string(dataset, "sift", "Dataset name, e.g. siftsmall");
DEFINE_string(result_path, "../baseline/nsg/result", "Result path");
DEFINE_int32(K, 3, "K");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string dataset = FLAGS_dataset;

  std::string source_path =
      "../data/" + dataset + "/" + dataset + "_base.fvecs";
  std::string query_path =
      "../data/" + dataset + "/" + dataset + "_query.fvecs";
  std::string gt_path =
      "../data/" + dataset + "/" + dataset + "_groundtruth.ivecs";
  std::string knn_path = FLAGS_result_path;
  uint32_t K = FLAGS_K;

  // Read data
  std::vector<std::vector<float>> data_set;
  uint32_t data_num, data_dim;
  ReadData(source_path, data_set, data_num, data_dim);

  // Read query
  std::vector<std::vector<float>> query_set;
  uint32_t query_num, query_dim;
  ReadData(query_path, query_set, query_num, query_dim);

  // Read ground truth
  uint32_t gt_num, gt_dim;
  std::vector<std::vector<uint32_t>> gt_set;
  ReadData(gt_path, gt_set, gt_num, gt_dim);

  // Read knn results
  uint32_t knn_num, knn_dim;
  std::vector<std::vector<uint32_t>> knn_results;
  ReadData(knn_path, knn_results, knn_num, knn_dim);

  // Calculate recall
  std::atomic<int> hit = 0;
  size_t dim = data_dim;
#pragma omp parallel for schedule(dynamic)
  for (uint32_t i = 0; i < query_num; ++i) {
    auto &knn = knn_results[i];
    std::vector<uint32_t> truth_knn;

    // fetch the top-K ground truth
    std::vector<std::pair<float, uint32_t>> knn_with_dist;
    for (auto gt : gt_set[i]) {
      knn_with_dist.emplace_back(std::make_pair(
          hnswlib::L2Sqr(query_set[i].data(), data_set[gt].data(), &dim),
          gt));
    }
    sort(knn_with_dist.begin(), knn_with_dist.end());
    truth_knn.clear();
    for (size_t j = 0; j < K; ++j) {
      truth_knn.emplace_back(knn_with_dist[j].second);
    }

    std::sort(knn.begin(), knn.end());
    std::sort(truth_knn.begin(), truth_knn.end());

    std::vector<uint32_t> intersection;
    std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(),
                          truth_knn.end(), std::back_inserter(intersection));
    hit.fetch_add(intersection.size());
  }

  double recall = static_cast<double>(hit.load()) / (query_num * K);
  std::cout << "Recall: " << recall << std::endl;

  return 0;
}