#include "core.h"
#include "include/strategy_include.h"
#include "util.h"
#include <cmath>
#include <csignal>
#include <gflags/gflags.h>
#include <unistd.h>

// 定义命令行参数
// DEFINE_string(dataset, "anton1m", "Dataset name, e.g. siftsmall");
DEFINE_string(dataset, "sift", "Dataset name, e.g. siftsmall");
// DEFINE_string(solve_strategy, "hnsw_v4_bin", "Solve strategy, e.g. hnsw");
DEFINE_string(solve_strategy, "hnsw_slim", "Solve strategy, e.g. hnsw");
DEFINE_int64(k, K, "Top-K nearest neighbors to search for");
DEFINE_int64(m, M, "Number of neighbors for each node in the index");
DEFINE_int64(m0, M0,
             "Number of neighbors for each node in the index at level 0");
// DEFINE_int64(ef_construction, EF_CONSTRUCTION,
//              "Maximum number of candidate neighbors considered during index "
//              "construction.");
// DEFINE_int64(ef_search, EF_SEARCH,
//              "Maximum number of candidates retained during the search
//              phase.");
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

DEFINE_int64(partial, 100, "build index with first a% data");

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

  float top_degree_percent0 = FLAGS_top_degree_percent0;
  float top_degree_percent = FLAGS_top_degree_percent;
  size_t top_M0 = FLAGS_top_M0;
  size_t low_m0 = FLAGS_low_m0;
  size_t top_M = FLAGS_top_M;
  size_t low_m = FLAGS_low_m;
  int threshold_level = FLAGS_threshold_level;
  int partial = FLAGS_partial;


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
  if (partial != 100) {
    suffix += "_" + std::to_string(partial);
  }
  suffix +=".graph";
  source_path =
      "../data/" + dataset + "/" + dataset + "_base.fvecs";
  query_path =
      "../data/" + dataset + "/" + dataset + "_query.fvecs";
  gt_path =
      "../data/" + dataset + "/" + dataset + "_groundtruth.ivecs";
  knn_path = "../statistics/knns/" + dataset + "_knn.ivecs";
  index_path = "../statistics/index/" + dataset + "/" + suffix;

  std::cout << "Index path: " << index_path << std::endl;

  SolveStrategy *strategy = nullptr;
  if (solve_strategy == "hnsw") {
    strategy = new HnswStrategy(source_path, query_path, index_path);
  } else if (solve_strategy == "hnsw_slim") {
    // strategy = new HnswSlimStrategy(source_path, query_path, index_path);
    if (partial == 100) {
      strategy = new HnswSlimStrategy(source_path, query_path, index_path,
                      threshold_level,
                      top_degree_percent0, top_degree_percent,
                      top_M0, low_m0, top_M, low_m);
    } else {
      strategy = new HnswSlimStrategy(source_path, query_path, index_path,
                  threshold_level,
                  top_degree_percent0, top_degree_percent,
                  top_M0, low_m0, top_M, low_m, partial);
    }
  } else {
    std::cout << "Unknown strategy: " << solve_strategy << std::endl;
    std::cout << "['hnsw', 'hnsw-slim', 'leann']"
              << std::endl;
    return 1;
  }



  // Processing
  strategy->solve();
  // std::cout << "Solve strategy: " + solve_strategy << std::endl;
  // // strategy->save_knn(knn_path);
  // // std::cout << "Save knn: " + knn_path << std::endl;
  // strategy->recall(gt_path);
  // std::cout << "Recall: " + gt_path << std::endl;
  delete strategy;
  return 0;
}
