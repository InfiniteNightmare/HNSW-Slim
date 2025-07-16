#include "core.h"
#include "include/strategy_include.h"
#include "util.h"
#include <cmath>
#include <csignal>
#include <gflags/gflags.h>
#include <oneapi/tbb/profiling.h>
#include <unistd.h>

DEFINE_string(dataset, "sift", "Dataset name, e.g. siftsmall");
DEFINE_string(solve_strategy, "hnsw_slim", "Solve strategy, e.g. hnsw");
DEFINE_int64(k, K, "Top-K nearest neighbors to search for");
DEFINE_int64(m, M, "Number of neighbors for each node in the index");
DEFINE_int64(m0, M0,
             "Number of neighbors for each node in the index at level 0");
DEFINE_int64(ef_construction, 128,
             "Maximum number of candidate neighbors considered during index "
             "construction.");
DEFINE_int64(ef_search, 128,
             "Maximum number of candidates retained during the search phase.");
DEFINE_string(branching_factor, BRANCHING_FACTOR,
              "Branching factor for the HNSW graph.");
DEFINE_int64(threshold_level, THRESHOLD_LEVEL,
             "Threshold level for the HNSW graph. This is used to control the "
             "number of levels in the HNSW graph.");

DEFINE_double(top_degree_percent0, 0.02f, "top a% nodes on level 0.");
DEFINE_double(top_degree_percent, 0.02f, "top a% nodes.");
DEFINE_int64(top_M0, 32, "neighbors");
DEFINE_int64(low_m0, 8, "neighbors");
DEFINE_int64(top_M, 16, "neighbors");
DEFINE_int64(low_m, 4, "neighbors");

DEFINE_int64(level_ratio, 50, "a% for level > 0");
DEFINE_int64(Mm_ratio, 25, "a% for level > 0");

void DatasetInfo(std::string dataset_path) {
  std::uint32_t num = 0, dim = 0;
  ReadSize(dataset_path, dim, num);
  std::cout << dataset_path << ", " << num << ", " << dim << std::endl;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string dataset = FLAGS_dataset;
  std::string solve_strategy = FLAGS_solve_strategy;
  K = FLAGS_k;
  M = FLAGS_m;
  M0 = FLAGS_m0;
  EF_CONSTRUCTION = FLAGS_ef_construction;
  EF_SEARCH = FLAGS_ef_search;
  BRANCHING_FACTOR = FLAGS_branching_factor;
  THRESHOLD_LEVEL = FLAGS_threshold_level;

  size_t level_ratio = FLAGS_level_ratio;
  size_t Mm_ratio = FLAGS_Mm_ratio;

  double ratio = 1.0 * level_ratio / 100.0;
  float top_degree_percent0 = FLAGS_top_degree_percent0;
  float top_degree_percent = top_degree_percent0;
  size_t top_M0 = FLAGS_top_M0;
  size_t low_m0 = top_M0 * Mm_ratio / 100;
  size_t top_M = ratio * top_M0;
  size_t low_m = ratio * low_m0;
  int threshold_level = FLAGS_threshold_level;

  // Initialization
  std::string source_path;
  std::string query_path;
  std::string gt_path;
  std::string knn_path;
  std::string index_path;

  // Create a filename for saving the index
  std::string suffix = solve_strategy + "_";
  suffix += std::to_string(EF_CONSTRUCTION) + "_";
  suffix += std::to_string(M) + "_";
  suffix += BRANCHING_FACTOR;
  std::string tmp_suffix = "";
  tmp_suffix += "_" + std::to_string(threshold_level);
  tmp_suffix += "_" + std::to_string(top_degree_percent0);
  tmp_suffix += "_" + std::to_string(top_degree_percent);
  tmp_suffix += "_" + std::to_string(top_M0);
  tmp_suffix += "_" + std::to_string(low_m0);
  tmp_suffix += "_" + std::to_string(top_M);
  tmp_suffix += "_" + std::to_string(low_m);

  suffix += tmp_suffix;
  suffix += ".graph";

  source_path = "../data/" + dataset + "/" + dataset + "_base.fvecs";
  query_path = "../data/" + dataset + "/" + dataset + "_query.fvecs";
  gt_path = "../data/" + dataset + "/" + dataset + "_groundtruth.ivecs";
  knn_path = "../statistics/knns/" + dataset + "_knn.ivecs";
  index_path = "../statistics/index/" + dataset + "/" + suffix;

  std::cout << "Index path: " << index_path << std::endl;
  std::cout << "gt path: " << gt_path << std::endl;

  std::cout << "Running with param: " << "alpha0%: " << top_degree_percent0
            << ", " << "alpha%: " << top_degree_percent << ", "
            << "top_m0: " << top_M0 << ", " << "top_m: " << top_M << ", "
            << "low_m0: " << low_m0 << ", " << "low_m: " << low_m << ", "
            << std::endl;

  SolveStrategy *strategy = nullptr;
  if (solve_strategy == "hnsw") {
    strategy = new HnswStrategy(source_path, query_path, index_path);
  } else if (solve_strategy == "hnsw_slim") {
    strategy =
        new HnswSlimStrategy(source_path, query_path, index_path, threshold_level,
                           top_degree_percent0, top_degree_percent, top_M0,
                           low_m0, top_M, low_m, 100, tmp_suffix);
  } else if (solve_strategy == "bruteforce") {
    strategy =
        new BruteForce(source_path, query_path, index_path, gt_path, 100);
  } else {
    std::cout << "Unknown strategy: " << solve_strategy << std::endl;
    std::cout << "['hnsw', 'hnsw_slim', 'bruteforce']"
              << std::endl;
    return 1;
  }

  strategy->solve();
  std::cout << "Solve strategy: " + solve_strategy << std::endl;
  strategy->recall(gt_path);
  std::cout << "Recall: " + gt_path << std::endl;
  delete strategy;
  return 0;
}
