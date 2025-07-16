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



void DatasetInfo() {
  std::string data_path = "../data";
  std::filesystem::path path = data_path;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_directory()) {
      std::filesystem::path rel_path = std::filesystem::relative(entry.path(), path);

      std::string dataset_path = data_path + "/" + rel_path.string()
                + "/" + rel_path.string() + "_base.fvecs";
      std::uint32_t num = 0, dim = 0;
      ReadSize(dataset_path, dim, num);
      std::cout << rel_path.string() << ", " << num << ", " << dim << std::endl;
    }
  }
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

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
  float top_degree_percent = ratio * top_degree_percent0;
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
  tmp_suffix += "_" + std::to_string(threshold_level) + "_";
  tmp_suffix += std::to_string(top_M0) + "_";
  tmp_suffix += std::to_string(level_ratio) + "_";
  tmp_suffix += std::to_string(Mm_ratio);

  suffix += tmp_suffix;

  suffix +=".graph";





  size_t out_m = 8;

  std::string dataset = "deep15m";
  std::string out_data = "deep" + std::to_string(out_m) + "m";
  source_path =
      "../data/" + dataset + "/" + dataset + "_base.fvecs";
  std::string out_path =
      "../data/" + out_data + "/" + out_data + "_base.fvecs";



  {
    std::cout << source_path << std::endl;
    std::cout << out_path << std::endl;


    uint32_t data_num_, data_dim_;
    std::vector<std::vector<float>> input_data;
    std::vector<std::vector<float>> output_data;


    ReadData(source_path, input_data, data_num_, data_dim_);

    for (size_t i=0; i<out_m * 1000000; i++) {
      output_data.emplace_back(input_data[i]);
    }

    WriteData(out_path, output_data);
  }
  uint32_t data_num_, data_dim_;
  std::vector<std::vector<float>> input_data;
  ReadData(out_path, input_data, data_num_, data_dim_);

  return 0;
}
