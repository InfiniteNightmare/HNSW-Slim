#include "hnswlib/hnswalg_slim.h"
#include "hnswlib/hnswlib.h"

// #define CPPHTTPLIB_ZLIB_SUPPORT
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
DEFINE_int64(branching_factor, 32,
              "Branching factor for the HNSW graph.");
DEFINE_int64(ef_search, 128,
             "Maximum number of candidates retained during the search phase.");
DEFINE_int64(partial, 100, "a% of data loaded");


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  httplib::Server svr;

  std::string dataset = FLAGS_dataset;
  int M = FLAGS_m;
  int branching_factor = FLAGS_branching_factor;
  int ef_construction = FLAGS_ef_construction;
  int ef_search = FLAGS_ef_search;
  int partial = FLAGS_partial;

  std::string source_path =
      "../data/" + dataset + "/" + dataset + "_base.fvecs";
  std::string hnsw_path =
        std::format("../statistics/index/{}/hnsw_{}_{}_{}.graph",
                    dataset, ef_construction, M, branching_factor);
  std::string index_path =
      std::format("../statistics/index/{}/hnsw_slim_{}_{}_{}.graph",
                  dataset, ef_construction, M, branching_factor);
  if (partial < 100) {
    hnsw_path.replace(hnsw_path.find(".graph"), 6, std::format("_{}.graph", partial));
    index_path.replace(index_path.find(".graph"), 6, std::format("_{}.graph", partial));
  }

  std::cout << hnsw_path << std::endl;
  std::cout << index_path << std::endl;

  uint32_t data_num, data_dim;
  {
    std::vector<std::vector<float>> data_set;
    ReadData(source_path, data_set, data_num, data_dim);
  }


  hnswlib::L2Space l2space(data_dim);
  hnswlib::HierarchicalNSW<float> hnsw(&l2space, hnsw_path);
  hnsw.setEf(ef_search);

  hnswlib::HierarchicalNSWSlim<float> hnsw_slim(&l2space, index_path, false, data_num);
  hnsw_slim.setEf(ef_search);

  int num_query = 0;
  double query_cost = 0;
  // query
  svr.Post("/query", [&](const httplib::Request &req, httplib::Response &res) {
    QueryRequest qreq;
    if (!qreq.ParseFromString(req.body)) {
      res.status = 400;
      res.set_content("Invalid protobuf", "text/plain");
      return;
    }
    auto s_query = std::chrono::system_clock::now();
    int k = qreq.k();
    std::vector<float> vec(qreq.vector().begin(), qreq.vector().end());

    std::vector<uint32_t>knn_results(k);
    hnsw_slim.searchKnn(vec.data(), k, knn_results.data());
    QueryResponse qresp;
    for (auto it : knn_results) {
      qresp.add_labels(it);
    }
    std::string out;
    qresp.SerializeToString(&out);
    res.set_content(out, "application/octet-stream");
    num_query++;
    auto e_query = std::chrono::system_clock::now();
    query_cost += std::chrono::duration<double>(e_query - s_query).count();

    if (num_query == 10000) {
      std::cout << "query cost server: " << query_cost << std::endl;
    }
  });
  // set ef_search
  svr.Post("/setEf", [&](const httplib::Request &req, httplib::Response &res) {
    SetEfRequest setef_req;
    if (!setef_req.ParseFromString(req.body)) {
      res.status = 400;
      res.set_content("Invalid protobuf", "text/plain");
      return;
    }
    int new_ef_search = setef_req.ef_search();
    hnsw_slim.setEf(new_ef_search);
    SetEfResponse setef_resp;
    setef_resp.set_status("success");
    setef_resp.set_new_ef_search(new_ef_search);
    std::string out;
    setef_resp.SerializeToString(&out);
    res.set_content(out, "application/octet-stream");
  });
  // update index
  svr.Post("/updateIndex",
           [&](const httplib::Request &req, httplib::Response &res) {
             std::cout << "update Index" << std::endl;
             UpdateIndexRequest update_req;
             if (!update_req.ParseFromString(req.body)) {
               res.status = 400;
               res.set_content("Invalid protobuf", "text/plain");
               return;
             }

             size_t l_limit = hnsw.cur_element_count;
             size_t h_limit = l_limit + update_req.vectors_size();

#pragma omp parallel for schedule(dynamic)
             for (int i = 0; i < update_req.vectors_size(); ++i) {
               const VectorData &vec_data = update_req.vectors(i);
               std::vector<float> vec(vec_data.vector().begin(),
                                      vec_data.vector().end());
               assert(vec_data.id() < h_limit && vec_data.id() >= l_limit);
               hnsw.addPoint(vec.data(), vec_data.id());
             }
             std::ostringstream oss(std::ios::binary);
             hnsw_slim.convertFromHNSWWithDiff(&hnsw, oss);

             std::string out = oss.str();
             res.set_content(out, "application/octet-stream");
             std::cout << "Fin UpdateIndex, stream size: " << out.size() << std::endl;
           });

  svr.listen("0.0.0.0", 8000);
  return 0;
}