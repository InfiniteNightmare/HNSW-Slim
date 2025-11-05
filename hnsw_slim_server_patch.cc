#include "hnswlib/hnswalg_slim.h"
#include "hnswlib/hnswlib.h"

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
DEFINE_int64(delete_rate, 5, "a% of data deleted");

void recall(std::string dataset, hnswlib::HierarchicalNSWSlim<float>& hnsw_slim,
            int ef_search) {

  hnsw_slim.setEf(ef_search);

  std::string query_path =
      "../data/" + dataset + "/" + dataset + "_query.fvecs";
  uint32_t query_num, query_dim;
  std::vector<std::vector<float>> query_set;
  ReadData(query_path, query_set, query_num, query_dim);
  std::vector<std::vector<uint32_t>> knn_results(query_num);

  for (uint32_t i = 0; i < query_num; ++i) {
    knn_results[i].resize(K);
  }
  auto s_solve = std::chrono::system_clock::now();
  double query_time = 0;

  // #pragma omp parallel for schedule(dynamic)
  for (uint32_t i = 0; i < query_num; ++i) {
    hnsw_slim.searchKnn(query_set[i].data(), K, knn_results[i].data());
  }
  auto e_solve = std::chrono::system_clock::now();
  std::cout << "solve cost: " << time_cost(s_solve, e_solve) << " (ms)\n";


  std::string source_path =
      "../data/" + dataset + "/" + dataset + "_base.fvecs";

  uint32_t data_num, data_dim;
  std::vector<std::vector<float>> data_set;
  ReadData(source_path, data_set, data_num, data_dim);

  int k = 3;

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
}




int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  httplib::Server svr;

  std::string dataset = FLAGS_dataset;
  int M = FLAGS_m;
  int branching_factor = FLAGS_branching_factor;
  int ef_construction = FLAGS_ef_construction;
  int ef_search = FLAGS_ef_search;
  int partial = FLAGS_partial;
  int delete_rate = FLAGS_delete_rate;

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
  hnswlib::HierarchicalNSW<float> hnsw(&l2space, hnsw_path, false, 0, true);
  hnsw.setEf(ef_search);

  hnswlib::HierarchicalNSWSlim<float> hnsw_slim(&l2space, index_path, false, data_num);
  hnsw_slim.setEf(ef_search);

  int num_query = 0;
  double query_cost = 0;

  size_t l_limit = 0;
  size_t h_limit = hnsw.cur_element_count;

  std::vector<std::pair<size_t, std::vector<float>>> deleted_nodes;

  bool inited = false;
  size_t changed_old_cnt, ind_old = 0;
  size_t changed_new_cnt, ind_new = 0;
  std::string meta;
  size_t patch_limit = 200 * 1024 * 1024;
  size_t cur_element_size;

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

  svr.Post("/updateIndex",
           [&](const httplib::Request &req, httplib::Response &res) {
     std::cout << "update Index" << std::endl;
     UpdateIndexRequest update_req;
     if (!update_req.ParseFromString(req.body)) {
       res.status = 400;
       res.set_content("Invalid protobuf", "text/plain");
       return;
     }
     l_limit += update_req.vectors_size();
     h_limit += update_req.vectors_size();

#pragma omp parallel for schedule(dynamic)
     for (int i = 0; i < update_req.vectors_size(); ++i) {
       const VectorData &vec_data = update_req.vectors(i);
       std::vector<float> vec(vec_data.vector().begin(),
                              vec_data.vector().end());
       if (vec_data.id() >= h_limit || vec_data.id() < l_limit) {
         std::string error_msg = std::format("error 161: {} < {} < {}",
                  l_limit, vec_data.id(), h_limit);
          std::cout << error_msg << std::endl;
       }
       assert(vec_data.id() < h_limit && vec_data.id() >= l_limit);
       hnsw.addPoint(vec.data(), vec_data.id(), false);
     }

     // delete_rate%
     int delete_limit = 0;
     for (int i = 0; i < update_req.vectors_size(); ++i) {
        if (delete_rate * i / 100 > delete_limit) {
          const VectorData &vec_data = update_req.vectors(i);
          std::vector<float> vec(vec_data.vector().begin(),
                                 vec_data.vector().end());
          hnsw.markDelete(vec_data.id());
          delete_limit = delete_rate * i / 100;
          deleted_nodes.emplace_back(std::make_pair(vec_data.id(), vec));
        }
     }

     std::ostringstream oss(std::ios::binary);
     hnsw_slim.convertFromHNSWWithDiff(&hnsw, oss);

     std::string out = oss.str();
     res.set_content(out, "application/octet-stream");
     std::cout << "Fin UpdateIndex, stream size: " << out.size() << std::endl;
   });



  svr.Post("/getLastBatch",
           [&](const httplib::Request &req, httplib::Response &res) {
     std::cout << "update Index" << std::endl;
     UpdateIndexRequest update_req;
     if (!update_req.ParseFromString(req.body)) {
       res.status = 400;
       res.set_content("Invalid protobuf", "text/plain");
       return;
     }

     auto s_solve = std::chrono::system_clock::now();

      if (!inited) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < deleted_nodes.size(); ++i) {
          hnsw.addPoint(deleted_nodes[i].second.data(),
                        deleted_nodes[i].first, true);
        }
        hnsw_slim.convertFromHNSWWithDiff(&hnsw, changed_old_cnt,
                                    changed_new_cnt);
        cur_element_size = hnsw_slim.cur_element_count_;

        inited = true;
      }

     std::ostringstream oss(std::ios::binary);
     size_t old_written = 0, new_written = 0;
     uint32_t finished = hnsw_slim.genPatch(oss, old_written,
            new_written, patch_limit, true);

     std::ostringstream oss_header(std::ios::binary);
     hnswlib::writeBinaryPOD(oss_header, finished);
     hnswlib::writeBinaryPOD(oss_header, cur_element_size);
     hnswlib::writeBinaryPOD(oss_header, old_written);
     hnswlib::writeBinaryPOD(oss_header, new_written);

     std::string out = oss_header.str() + oss.str();
     res.set_content(out, "application/octet-stream");
     std::cout << "Fin Last Patch, stream size: " << out.size() << std::endl;
     std::cout << "Fin Last Patch, num nodes: " << deleted_nodes.size() << std::endl;
     auto e_solve = std::chrono::system_clock::now();
     std::cout << "Last Patch cost: " << time_cost(s_solve, e_solve) << " (ms)\n";

   });

  svr.listen("0.0.0.0", 8000);
  return 0;
}