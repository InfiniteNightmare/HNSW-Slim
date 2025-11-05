#include "hnswlib/hnswlib.h"
#include "httplib.h"
#include "query.pb.h"
#include "util.h"
#include <gflags/gflags.h>
#include <iostream>
#include <vector>

DEFINE_string(dataset, "sift", "Dataset name, e.g. siftsmall");
DEFINE_int64(ef_search, 128,
             "Maximum number of candidates retained during the search phase.");
DEFINE_int64(k, 3, "Number of nearest neighbors to return");

double req_cost;
double serial_cost;
double post_cost;
double parse_cost;

bool query(httplib::Client &cli, const std::vector<float> &vec, int k,
           std::vector<uint32_t> &knn) {
  auto s_req = std::chrono::system_clock::now();
  QueryRequest req;
  req.set_k(k);
  for (float v : vec)
    req.add_vector(v);
  auto e_req = std::chrono::system_clock::now();
  req_cost += std::chrono::duration<double>(e_req - s_req).count();

  auto s_serial = std::chrono::system_clock::now();
  std::string body;
  req.SerializeToString(&body);
  auto e_serial = std::chrono::system_clock::now();
  serial_cost += std::chrono::duration<double>(e_serial - s_serial).count();

  auto s_post = std::chrono::system_clock::now();
  auto res = cli.Post("/query", body, "application/octet-stream");
  auto e_post = std::chrono::system_clock::now();
  post_cost += std::chrono::duration<double>(e_post - s_post).count();

  if (res && res->status == 200) {
    auto s_parse = std::chrono::system_clock::now();
    QueryResponse resp;
    if (resp.ParseFromString(res->body)) {
      knn.insert(knn.end(), resp.labels().begin(), resp.labels().end());
      while (knn.size() < k) {
        knn.emplace_back(-1);
      }
      auto e_parse = std::chrono::system_clock::now();
      parse_cost += std::chrono::duration<double>(e_parse - s_parse).count();
      return true;
    } else {
      std::cerr << "Failed to parse response protobuf" << std::endl;
    }
  } else {
    std::cerr << "Request failed: " << (res ? res->status : 0) << std::endl;
    if (res)
      std::cerr << res->body << std::endl;
  }
  return false;
}

bool setEf(httplib::Client &cli, int new_ef_search) {
  SetEfRequest setef_req;
  setef_req.set_ef_search(new_ef_search);
  std::string setef_body;
  setef_req.SerializeToString(&setef_body);
  auto setef_res = cli.Post("/setEf", setef_body, "application/octet-stream");
  if (setef_res && setef_res->status == 200) {
    SetEfResponse setef_resp;
    if (setef_resp.ParseFromString(setef_res->body)) {
      std::cout << "setEf status: " << setef_resp.status()
                << ", new_ef_search: " << setef_resp.new_ef_search()
                << std::endl;
      return true;
    } else {
      std::cerr << "Failed to parse setEf response protobuf" << std::endl;
    }
  } else {
    std::cerr << "setEf request failed: " << (setef_res ? setef_res->status : 0)
              << std::endl;
    if (setef_res)
      std::cerr << setef_res->body << std::endl;
  }
  return false;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  const char *host = "HOST_IP";
  int port = 8000;
  httplib::Client cli(host, port);

  req_cost = 0;
  serial_cost = 0;
  post_cost = 0;
  parse_cost = 0;

  std::string dataset = FLAGS_dataset;
  int ef_search = FLAGS_ef_search;
  int k = FLAGS_k;

  // Read dataset
  std::string source_path =
      "../data/" + dataset + "/" + dataset + "_base.fvecs";
  uint32_t data_num, data_dim;
  std::vector<std::vector<float>> data_set;
  ReadData(source_path, data_set, data_num, data_dim);

  // set ef
  if (!setEf(cli, ef_search)) {
    std::cerr << "Failed to set ef_search" << std::endl;
    return 1;
  }

  // query
  std::string query_path =
      "../data/" + dataset + "/" + dataset + "_query.fvecs";
  uint32_t query_num, query_dim;
  std::vector<std::vector<float>> query_set;
  ReadData(query_path, query_set, query_num, query_dim);

  // Solve query
  std::vector<std::vector<uint32_t>> knn_results(query_num);
  auto s_solve = std::chrono::system_clock::now();
  for (uint32_t i = 0; i < query_num; ++i) {
    if (!query(cli, query_set[i], k, knn_results[i])) {
      std::cerr << "Query failed for index " << i << std::endl;
      return 1;
    }
  }
  auto e_solve = std::chrono::system_clock::now();
  std::cout << "req cost: " << req_cost << "(s)" << std::endl;
  std::cout << "serial cost: " << serial_cost << "(s)" << std::endl;
  std::cout << "post cost: " << post_cost << "(s)" << std::endl;
  std::cout << "parse cost: " << parse_cost << "(s)" << std::endl;

  std::cout << "\n\nsolve cost: "
            << std::chrono::duration<double>(e_solve - s_solve).count() << "(s)"
            << std::endl;

  // Recall calculation
  std::string gt_path =
      "../data/" + dataset + "/" + dataset + "_groundtruth.ivecs";
  uint32_t gt_num, gt_dim;
  std::vector<std::vector<uint32_t>> gt_set;
  ReadData(gt_path, gt_set, gt_num, gt_dim);
  std::atomic<int> hit = 0;
  size_t dim = query_dim;
#pragma omp parallel for schedule(dynamic)
  for (uint32_t i = 0; i < query_num; ++i) {
    auto &knn = knn_results[i];
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
  std::cout << "Recall: " << recall << std::endl;
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
