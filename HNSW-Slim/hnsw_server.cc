#include "hnswlib/hnswlib.h"
//#define CPPHTTPLIB_ZLIB_SUPPORT
#include "httplib.h"
#include "query.pb.h"
#include "util.h"
#include <format>
#include <gflags/gflags.h>
#include <string>

DEFINE_string(dataset, "sift", "Dataset name, e.g. siftsmall");
DEFINE_int64(m, 30, "Number of neighbors for each node in the index");
DEFINE_int64(ef_construction, 128,
             "Maximum number of candidate neighbors considered during index "
             "construction.");
DEFINE_int64(ef_search, 128,
             "Maximum number of candidates retained during the search phase.");

DEFINE_int64(branching_factor, 32,
              "Branching factor for the HNSW graph.");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  httplib::Server svr;

  std::string dataset = FLAGS_dataset;
  int M = FLAGS_m;
  int ef_construction = FLAGS_ef_construction;
  int ef_search = FLAGS_ef_search;
  int branching_factor = FLAGS_branching_factor;
  std::string source_path =
      "../HNSW/data/" + dataset + "/" + dataset + "_base.fvecs";
  std::string index_path =
      std::format("../statistics/index/{}/hnsw_{}_{}_{}.graph",
                  dataset, ef_construction, M, branching_factor);
  uint32_t data_dim;
  ReadData(source_path, data_dim);
  hnswlib::L2Space l2space(data_dim);
  hnswlib::HierarchicalNSW<float> hnsw(&l2space, index_path);
  hnsw.setEf(ef_search);

  int64_t solve_time = 0;
  double solve_cost = 0;
  double search_cost = 0;
  double parse_cost = 0;
  double resp_cost = 0;
  double serial_cost = 0;

  int solve_num = 0;
  int query_num = 10000;

  // Query
  svr.Post("/query", [&](const httplib::Request &req, httplib::Response &res) {
    QueryRequest qreq;
    auto s_parse = std::chrono::system_clock::now();
    if (!qreq.ParseFromString(req.body)) {
      res.status = 400;
      res.set_content("Invalid protobuf", "text/plain");
      return;
    }
    auto e_parse = std::chrono::system_clock::now();
    parse_cost += std::chrono::duration<double>(e_parse - s_parse).count();


    auto s_query = std::chrono::system_clock::now();
    auto s_search = std::chrono::system_clock::now();
    int k = qreq.k();
    std::vector<float> vec(qreq.vector().begin(), qreq.vector().end());
    auto result = hnsw.searchKnn(vec.data(), k);
    auto e_search = std::chrono::system_clock::now();
    search_cost += std::chrono::duration<double>(e_search - s_search).count();


    auto s_resp = std::chrono::system_clock::now();
    QueryResponse qresp;
    while (!result.empty() && qresp.labels_size() < k) {
      auto top = result.top();
      qresp.add_labels(top.second);
      qresp.add_distances(top.first);
      result.pop();
    }
    auto e_resp = std::chrono::system_clock::now();
    resp_cost += std::chrono::duration<double>(e_resp - s_resp).count();


    auto s_serial = std::chrono::system_clock::now();
    std::string out;
    qresp.SerializeToString(&out);
    res.set_content(out, "application/octet-stream");
    auto e_serial = std::chrono::system_clock::now();
    serial_cost += std::chrono::duration<double>(e_serial - s_serial).count();


    auto e_query = std::chrono::system_clock::now();
    solve_cost += std::chrono::duration<double>(e_query - s_query).count();
    solve_time += time_cost(s_query, e_query);
  });
  // Set ef_search
  svr.Post("/setEf", [&](const httplib::Request &req, httplib::Response &res) {
    SetEfRequest setef_req;
    if (!setef_req.ParseFromString(req.body)) {
      res.status = 400;
      res.set_content("Invalid protobuf", "text/plain");
      return;
    }
    int new_ef_search = setef_req.ef_search();
    hnsw.setEf(new_ef_search);
    SetEfResponse setef_resp;
    setef_resp.set_status("success");
    setef_resp.set_new_ef_search(new_ef_search);
    std::string out;
    setef_resp.SerializeToString(&out);
    res.set_content(out, "application/octet-stream");
  });

  svr.listen("0.0.0.0", 8000);
  return 0;
}