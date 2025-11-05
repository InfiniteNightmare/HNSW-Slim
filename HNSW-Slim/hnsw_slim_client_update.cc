// #define CPPHTTPLIB_ZLIB_SUPPORT
#include "httplib.h"

#include "query.pb.h"
#include "util.h"
#include "hnswlib/hnswalg_slim.h"
#include <cstdint>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <sstream>
#include <vector>

DEFINE_string(dataset, "sift", "Dataset name, e.g. siftsmall");
DEFINE_int64(m, 30, "Number of neighbors for each node in the index");
DEFINE_int64(ef_construction, 128,
             "Maximum number of candidate neighbors considered during index "
             "construction.");
DEFINE_int64(branching_factor, 4,
              "Branching factor for the HNSW graph.");
DEFINE_int64(partial, 90, "a% loaded");
DEFINE_int64(update_size, 10000, "size of each update");

bool updateIndex(httplib::Client &cli,
                 std::vector<std::pair<uint32_t, std::vector<float>>> &data,
                 hnswlib::HierarchicalNSWSlim<float>& hnsw_slim) {
  UpdateIndexRequest req;
  for (const auto &item : data) {
    VectorData *vec_data = req.add_vectors();
    vec_data->set_id(item.first);
    for (float v : item.second) {
      vec_data->add_vector(v);
    }
  }
  std::string req_body;
  req.SerializeToString(&req_body);
  auto res = cli.Post("/updateIndex", req_body, "application/octet-stream");
  if (res && res->status == 200) {
    std::istringstream in(res->body, std::ios::binary);
    hnsw_slim.patchFromStream(in);
    return true;
  } else {
    std::cerr << "updateIndex failed: " << (res ? res->status : 0) << std::endl;
    if (res)
      std::cerr << res->body << std::endl;
  }
  return false;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  const char *host = "HOST_IP";
  int port = 8000;
  httplib::Client cli(host, port);

  std::string dataset = FLAGS_dataset;
  int M = FLAGS_m;
  int ef_construction = FLAGS_ef_construction;
  int branching_factor = FLAGS_branching_factor;
  int partial = FLAGS_partial;
  int update_size = FLAGS_update_size;

  // Read dataset
  std::string source_path =
      "../data/" + dataset + "/" + dataset + "_base.fvecs";

  std::string index_path =
        std::format("../statistics/index/{}/hnsw_slim_{}_{}_{}_{}.graph",
                    dataset, ef_construction, M, branching_factor, partial);
  if (partial == 100) {
    index_path =
      std::format("../statistics/index/{}/hnsw_slim_{}_{}_{}.graph",
                  dataset, ef_construction, M, branching_factor);
  }

  uint32_t data_num, data_dim;
  std::vector<std::vector<float>> data_set;
  ReadData(source_path, data_set, data_num, data_dim);
  hnswlib::L2Space l2space(data_dim);
  hnswlib::HierarchicalNSWSlim<float> hnsw_slim(&l2space, index_path, false, data_num);

  cli.set_compress(true);
  cli.set_decompress(false);
  uint32_t loaded_num = data_num * partial / 100;
  for (; loaded_num < data_num; loaded_num += update_size) {
    auto s_batch = std::chrono::system_clock::now();

    std::vector<std::pair<uint32_t, std::vector<float>>> update_data;
    for (int i=0; i<update_size; i++) {
      update_data.emplace_back(std::make_pair(loaded_num+i, data_set[loaded_num+i]));
    }
    if (!updateIndex(cli, update_data, hnsw_slim)) {
      std::cerr << "Failed to update index" << std::endl;
      return 1;
    }
    auto e_batch = std::chrono::system_clock::now();

    std::cout << "update index: " << time_cost(s_batch, e_batch) << " (ms)\n";
  }

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
