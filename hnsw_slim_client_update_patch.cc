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
DEFINE_int64(k, 5, "Number of nearest neighbors to return");
DEFINE_int64(partial, 90, "a% loaded");
DEFINE_int64(update_size, 10000, "size of each update");
DEFINE_int64(ef_search, 128,
             "Maximum number of candidates retained during the search phase.");

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


bool updateIndex(httplib::Client &cli,
                std::unordered_map<uint32_t, std::vector<float>> &data,
                 // std::vector<std::pair<uint32_t, std::vector<float>>> &data,
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
    hnsw_slim.patchFromStream(in, data);
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
  int k = FLAGS_k;
  int M = FLAGS_m;
  int ef_construction = FLAGS_ef_construction;
  int branching_factor = FLAGS_branching_factor;
  int partial = FLAGS_partial;
  int update_size = FLAGS_update_size;
  int ef_search = FLAGS_ef_search;

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

  std::string sta_path = "../patch_log_start.txt";
  std::string end_path = "../patch_log_end.txt";

  uint32_t data_num, data_dim;
  std::vector<std::vector<float>> read_data;
  std::ifstream in(source_path, std::ios::binary);
  {
    if (!in.is_open()) {
      std::cout << "open file error" << std::endl;
      exit(-1);
    }
    in.read((char *)&data_dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    data_num = (uint32_t)(fsize / (data_dim + 1) / 4);
    read_data.resize(data_num / 100);
    for (uint32_t i = 0; i < data_num / 100; ++i) {
      read_data[i].resize(data_dim);
    }
    in.seekg(0, std::ios::beg);
    in.seekg(data_num / 100 * (4 + data_dim * 4), std::ios::cur);
  }

  hnswlib::L2Space l2space(data_dim);
  hnswlib::HierarchicalNSWSlim<float> hnsw_slim(&l2space, index_path, false, data_num);

  cli.set_read_timeout(6000);
  cli.set_compress(true);
  cli.set_decompress(false);

  // add 1% data
  for (uint32_t i = 1; i < 100; i++) {
    std::unordered_map<uint32_t, std::vector<float>> update_data;

    for (uint32_t j = 0; j < data_num / 100; j++) {
      in.seekg(4, std::ios::cur);
      in.read((char *)read_data[j].data(), data_dim * 4);
    }

    uint32_t sta = data_num * i / 100;
    for (uint32_t j = 0; j < data_num  / 100; j++) {
      update_data[sta + j] = read_data[j];
    }
    std::cout << "\t" << i << "/100" << std::endl;
    if (!updateIndex(cli, update_data, hnsw_slim)) {
      std::cerr << "Failed to update index" << std::endl;
      return 1;
    }
  }
  in.close();

  std::unordered_map<uint32_t, std::vector<float>> update_data;
  update_data[0] = read_data[0];
  UpdateIndexRequest req;
  for (const auto &item : update_data) {
    VectorData *vec_data = req.add_vectors();
    vec_data->set_id(item.first);
    for (float v : item.second) {
      vec_data->add_vector(v);
    }
  }

  std::string req_body;
  req.SerializeToString(&req_body);

  uint32_t finished = 0;
  while(finished == 0) {
    auto res = cli.Post("/getLastBatch", req_body, "application/octet-stream");
    if (res && res->status == 200) {
      std::cout << "last batch" << std::endl;
      std::istringstream in(res->body, std::ios::binary);

      hnswlib::readBinaryPOD(in, finished);
      hnsw_slim.patchFromStream(in, true);
    } else {
      std::cerr << "lastPatch failed: " << (res ? res->status : 0) << std::endl;
      if (res)
        std::cerr << res->body << std::endl;
    }
  }

  std::vector<std::vector<float>> data_set;
  ReadData(source_path, data_set, data_num, data_dim);
  std::cout << "fin" << std::endl;

  std::string query_path =
      "../data/" + dataset + "/" + dataset + "_query.fvecs";

  uint32_t query_num, query_dim;
  std::vector<std::vector<float>> query_set;
  ReadData(query_path, query_set, query_num, query_dim);
  std::vector<std::vector<uint32_t>> knn_results(query_num);
  std::cout << "[query]:\t" << query_num << ", " << query_dim << std::endl;

   // Recall calculation
   std::string gt_path =
       "../data/" + dataset + "/" + dataset + "_groundtruth.ivecs";
   uint32_t gt_num, gt_dim;
   std::vector<std::vector<uint32_t>> gt_set;
   ReadData(gt_path, gt_set, gt_num, gt_dim);
   std::cout << "[gt]:\t" << gt_num << ", " << gt_dim << std::endl;

   std::atomic<int> hit = 0;

  knn_results.resize(query_num);
   for (uint32_t i = 0; i < query_num; ++i) {
     knn_results[i].resize(k);
   }

  hnsw_slim.setEf(ef_search);
   auto s_solve = std::chrono::system_clock::now();

   #pragma omp parallel for schedule(dynamic)
   for (uint32_t i = 0; i < query_num; ++i) {
     hnsw_slim.searchKnn(query_set[i].data(), k, knn_results[i].data());
   }
   auto e_solve = std::chrono::system_clock::now();
   std::cout << "solve cost: " << time_cost(s_solve, e_solve) << " (ms)\n";

  {
    size_t dim = query_dim;
   #pragma omp parallel for schedule(dynamic)
     for (uint32_t i = 0; i < query_num; ++i) {
       auto &knn = knn_results[i];
       // auto& truth_knn = gt_set[i];
       std::vector<uint32_t> truth_knn;

       // fetch the top-K ground truth
       std::vector<std::pair<float, uint32_t>> knn_with_dist;
       for (auto gt : gt_set[i]) {
         knn_with_dist.emplace_back(
             hnswlib::L2Sqr(query_set[i].data(), data_set[gt].data(), &dim), gt);
       }
       sort(knn_with_dist.begin(), knn_with_dist.end());
       truth_knn.clear();
       for (size_t j = 0; j < k; ++j) {
         truth_knn.emplace_back(knn_with_dist[j].second);
       }

       std::sort(knn.begin(), knn.end());
       std::sort(truth_knn.begin(), truth_knn.end());

       std::vector<uint32_t> intersection;
       std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(),
                             truth_knn.end(), std::back_inserter(intersection));
       hit.fetch_add(intersection.size());
     }

     float recall = static_cast<float>(hit.load()) / (query_num * k);
      std::cout << "recall\t" << recall << std::endl;
     }

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
