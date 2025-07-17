#pragma once

#include "core.h"
#include "hnswlib.h"
#include "hnswlib/hnswalg.h"
#include "visited_list_pool.h"
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cstdlib>
#include <folly/concurrency/container/atomic_grow_array.h>
#include <limits>
#include <memory>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;
typedef int levelsizeint;
typedef uint16_t offsetint;

template <typename dist_t>
class HierarchicalNSWSlim : public AlgorithmInterface<dist_t> {
public:
  static const unsigned char DELETE_MARK = 0x01;

  size_t max_elements_{0};
  size_t cur_element_count_{0}; // current number of elements
  size_t size_data_per_element_{0};
  bool has_deleted_elements_{
      false}; // flag to indicate if there are deleted elements
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  int maxlevel_{0};
  int threshold_level_{0};

  std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

  tableint enterpoint_node_{0};

  size_t offsetTotalNeighbor_{0}, offsetData_{0}, offsetNeighbor_{0},
      label_offset_{0};

  char *elements_{nullptr};

  size_t data_size_{0};

  DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_{nullptr};

  std::unordered_map<labeltype, tableint> label_lookup_;

  float top_degree_percent0_{0.02f}; // \alpha_0\%
  float top_degree_percent_{0.02f};  // \alpha\%
  size_t top_degree_M0_{32};         // M_{h_0}
  size_t low_degree_m0_{8};          // M_{l_0}
  size_t top_degree_M_{16};          // M_{h}
  size_t low_degree_m_{4};           // M_{l}

  mutable std::atomic<long> metric_distance_computations{0};
  mutable std::atomic<long> metric_hops{0};

  HierarchicalNSWSlim(SpaceInterface<dist_t> *s) {}

  HierarchicalNSWSlim(SpaceInterface<dist_t> *s, const std::string &location,
                     bool nmslib = false, size_t max_elements = 0,
                     bool allow_replace_deleted = false) {
    loadIndex(location, s, max_elements);
  }

  HierarchicalNSWSlim(SpaceInterface<dist_t> *s, size_t max_elements,
                     size_t M = 16, size_t ef_construction = 200,
                     size_t threshold_level = 0,
                     float top_degree_percent0 = 0.02f,
                     float top_degree_percent = 0.02f,
                     size_t top_degree_M0 = 32, size_t low_degree_m0 = 8,
                     size_t top_degree_M = 16, size_t low_degree_m = 4,
                     size_t random_seed = 100,
                     bool allow_replace_deleted = false) {
    max_elements_ = max_elements;
    has_deleted_elements_ = false;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    if (M <= 10000) {
      M_ = M;
    } else {
      HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse "
                 "effects."
              << std::endl;
      HNSWERR << "         Cap to 10000 will be applied for the rest of the "
                 "processing."
              << std::endl;
      M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    threshold_level_ = threshold_level;
    top_degree_percent0_ = top_degree_percent0;
    top_degree_percent_ = top_degree_percent;
    top_degree_M0_ = top_degree_M0;
    low_degree_m0_ = low_degree_m0;
    top_degree_M_ = top_degree_M;
    low_degree_m_ = low_degree_m;

    offsetTotalNeighbor_ = sizeof(levelsizeint);
    label_offset_ = offsetTotalNeighbor_ + sizeof(linklistsizeint);
    offsetNeighbor_ = label_offset_ + sizeof(labeltype);
    offsetData_ = offsetNeighbor_ + sizeof(char **);
    size_data_per_element_ = offsetData_ + data_size_;

    elements_ = (char *)malloc(max_elements_ * size_data_per_element_);

    cur_element_count_ = 0;

    visited_list_pool_ =
        std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements_));

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;
  }

  HierarchicalNSWSlim(HierarchicalNSW<dist_t> *hnsw) { convertFromHNSW(hnsw); }

  ~HierarchicalNSWSlim() { clear(); }

  void addPoint(const void *datapoint, labeltype label,
                bool replace_deleted = false) {
    throw std::runtime_error("HierarchicalNSWSlim does not support addPoint");
  }

  void clear() {
    if (elements_) {
      for (size_t i = 0; i < cur_element_count_; i++) {
        char *neighbors = get_neighbors(i);
        if (neighbors) {
          free(neighbors);
        }
      }
      free(elements_);
      elements_ = nullptr;
    }
    cur_element_count_ = 0;
    visited_list_pool_.reset(nullptr);
  }

  struct {
    constexpr bool
    operator()(std::pair<dist_t, tableint> const &a,
               std::pair<dist_t, tableint> const &b) const noexcept {
      return a.first < b.first;
    }
  } compare_by_first;

  struct {
    constexpr bool
    operator()(std::pair<dist_t, tableint> const &a,
               std::pair<dist_t, tableint> const &b) const noexcept {
      return a.first > b.first;
    }
  } compare_by_first_rev;

  struct CompareByFirst {
    constexpr bool
    operator()(std::pair<dist_t, tableint> const &a,
               std::pair<dist_t, tableint> const &b) const noexcept {
      return a.first > b.first;
    }
  };

  void setEf(size_t ef) { ef_ = ef; }

  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label,
           (elements_ + internal_id * size_data_per_element_ + label_offset_),
           sizeof(labeltype));
    return return_label;
  }

  inline labeltype *getExternalLabeLp(tableint internal_id) const {
    return (labeltype *)(elements_ + internal_id * size_data_per_element_ +
                         label_offset_);
  }

  inline char *getDataByInternalId(tableint internal_id) const {
    return (elements_ + internal_id * size_data_per_element_ + offsetData_);
  }

  size_t getMaxElements() { return max_elements_; }

  size_t getCurrentElementCount() { return cur_element_count_; }

  void searchBaseLayer(const void *data_point, int layer,
                       std::pair<dist_t, tableint> *top_candidates,
                       size_t &top_candidates_size,
                       std::pair<dist_t, tableint> *candidateSet, size_t ef,
                       vl_type *visited_array, vl_type visited_array_tag,
                       dist_t &lowerBound) const {
    size_t candidateSetSize = top_candidates_size;
    memcpy(candidateSet, top_candidates,
           top_candidates_size * sizeof(std::pair<dist_t, tableint>));

    std::make_heap(candidateSet, candidateSet + candidateSetSize,
                   compare_by_first_rev);

    while (candidateSetSize > 0) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet[0];
      if ((curr_el_pair.first) > lowerBound && top_candidates_size == ef) {
        break;
      }
      std::pop_heap(candidateSet, candidateSet + candidateSetSize--,
                    compare_by_first_rev);

      tableint curNodeNum = curr_el_pair.second;

      char *element = elements_ + curNodeNum * size_data_per_element_;
      char *neighbors = get_neighbors(element);
      if (neighbors == nullptr) {
        continue;
      }
      levelsizeint element_level = get_element_level(element);
      assert(element_level >= layer);
      offsetint offset = get_neighbor_offset_at_level(neighbors, layer);
      size_t size =
          (layer == element_level ? get_total_neighbor(element)
                                  : ((offsetint *)(neighbors))[layer]) -
          offset;
      if (size == 0)
        continue;
      tableint *data =
          (tableint *)(neighbors + sizeof(offsetint) * element_level) + offset;

#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *data), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *data + 64), _MM_HINT_T0);
      if (size > 1)
        _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);
      if (size > 2)
        _mm_prefetch(getDataByInternalId(*(data + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(data + j);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
        if (j + 1 < size)
          _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag)
          continue;
        visited_array[candidate_id] = visited_array_tag;
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates_size < ef || lowerBound > dist1) {
          candidateSet[candidateSetSize++] =
              std::make_pair(dist1, candidate_id);
          std::push_heap(candidateSet, candidateSet + candidateSetSize,
                         compare_by_first_rev);
          assert(get_element_level(elements_ +
                                   candidate_id * size_data_per_element_) >=
                 layer);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet[0].second),
                       _MM_HINT_T0);
#endif

          if (!isMarkedDeleted(candidate_id)) {
            top_candidates[top_candidates_size++] =
                std::make_pair(dist1, candidate_id);
            std::push_heap(top_candidates, top_candidates + top_candidates_size,
                           compare_by_first);
          }

          if (top_candidates_size > ef) {
            std::pop_heap(top_candidates,
                          top_candidates + top_candidates_size--,
                          compare_by_first);
          }

          if (top_candidates_size > 0) {
            lowerBound = top_candidates[0].first;
          }
        }
      }
    }
  }

  // bare_bone_search means there is no check for deletions and stop condition
  // is ignored in return of extra performance
  template <bool bare_bone_search = true, bool collect_metrics = false>
  void searchBaseLayerST(
      const void *data_point, std::pair<dist_t, tableint> *top_candidates,
      size_t &top_candidates_size, std::pair<dist_t, tableint> *candidate_set,
      size_t ef, vl_type *visited_array, vl_type visited_array_tag,
      dist_t &lowerBound = std::numeric_limits<dist_t>::max(),
      BaseSearchStopCondition<dist_t> *stop_condition = nullptr) const {
    size_t candidate_set_size = top_candidates_size;
    memcpy(candidate_set, top_candidates,
           top_candidates_size * sizeof(std::pair<dist_t, tableint>));

    std::make_heap(candidate_set, candidate_set + candidate_set_size,
                   compare_by_first_rev);

    while (candidate_set_size > 0) {
      std::pair<dist_t, tableint> current_node_pair = candidate_set[0];
      dist_t candidate_dist = current_node_pair.first;

      bool flag_stop_search;
      if (bare_bone_search) {
        flag_stop_search = candidate_dist > lowerBound;
      } else {
        if (stop_condition) {
          flag_stop_search =
              stop_condition->should_stop_search(candidate_dist, lowerBound);
        } else {
          flag_stop_search =
              candidate_dist > lowerBound && top_candidates_size == ef;
        }
      }
      if (flag_stop_search) {
        break;
      }
      std::pop_heap(candidate_set, candidate_set + candidate_set_size--,
                    compare_by_first_rev);

      tableint current_node_id = current_node_pair.second;

      char *element = elements_ + current_node_id * size_data_per_element_;
      char *neighbors = get_neighbors(element);
      if (neighbors == nullptr) {
        continue;
      }
      levelsizeint element_level = get_element_level(element);
      size_t size = element_level == 0 ? get_total_neighbor(element)
                                       : ((offsetint *)(neighbors))[0];
      if (size == 0)
        continue;
      tableint *data =
          (tableint *)(neighbors + sizeof(offsetint) * element_level);

      //                bool cur_node_deleted =
      //                isMarkedDeleted(current_node_id);
      if (collect_metrics) {
        metric_hops++;
        metric_distance_computations += size;
      }

#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *data), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *data + 64), _MM_HINT_T0);
      _mm_prefetch((char *)getDataByInternalId(*data), _MM_HINT_T0);
      _mm_prefetch((char *)(data + 1), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        int candidate_id = data[j];
#ifdef USE_SSE
        if (j + 1 < size) {
          _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
          _mm_prefetch(getDataByInternalId(*(data + j + 1)),
                       _MM_HINT_T0); ////////////
        }
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          char *currObj1 = (getDataByInternalId(candidate_id));
          dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

          bool flag_consider_candidate;
          if (!bare_bone_search && stop_condition) {
            flag_consider_candidate =
                stop_condition->should_consider_candidate(dist, lowerBound);
          } else {
            flag_consider_candidate =
                top_candidates_size < ef || lowerBound > dist;
          }

          if (flag_consider_candidate) {
            candidate_set[candidate_set_size++] =
                std::make_pair(dist, candidate_id);
            std::push_heap(candidate_set, candidate_set + candidate_set_size,
                           compare_by_first_rev);
#ifdef USE_SSE
            _mm_prefetch(
                getDataByInternalId(candidate_set[0].second), ///////////
                _MM_HINT_T0); ////////////////////////
#endif

            if (bare_bone_search || !isMarkedDeleted(candidate_id)) {
              top_candidates[top_candidates_size++] =
                  std::make_pair(dist, candidate_id);
              std::push_heap(top_candidates,
                             top_candidates + top_candidates_size,
                             compare_by_first);
              if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(
                    getExternalLabel(candidate_id), currObj1, dist);
              }
            }

            bool flag_remove_extra = false;
            if (!bare_bone_search && stop_condition) {
              flag_remove_extra = stop_condition->should_remove_extra();
            } else {
              flag_remove_extra = top_candidates_size > ef;
            }
            while (flag_remove_extra) {
              tableint id = top_candidates[0].second;
              std::pop_heap(top_candidates,
                            top_candidates + top_candidates_size--,
                            compare_by_first);
              if (!bare_bone_search && stop_condition) {
                stop_condition->remove_point_from_result(
                    getExternalLabel(id), getDataByInternalId(id), dist);
                flag_remove_extra = stop_condition->should_remove_extra();
              } else {
                flag_remove_extra = top_candidates_size > ef;
              }
            }

            if (top_candidates_size > 0) {
              lowerBound = top_candidates[0].first;
            }
          }
        }
      }
    }
    // free(candidate_set);
  }

  // bare_bone_search means there is no check for deletions and stop
  // condition is ignored in return of extra performance
  template <bool bare_bone_search = true, bool collect_metrics = false>
  void searchBaseLayerST(
      const void *data_point, std::pair<dist_t, tableint> *top_candidates,
      size_t &top_candidates_size, std::pair<dist_t, tableint> *candidate_set,
      size_t ef, vl_type *visited_array, vl_type visited_array_tag,
      BaseFilterFunctor *isIdAllowed = nullptr,
      BaseSearchStopCondition<dist_t> *stop_condition = nullptr) const {
    size_t candidate_set_size = 0;

    std::make_heap(candidate_set, candidate_set + candidate_set_size,
                   compare_by_first_rev);

    dist_t lowerBound = std::numeric_limits<dist_t>::max();
    bool has_valid_candidates = false;
    for (size_t i = 0; i < top_candidates_size; i++) {
      auto [dist, cand_id] = top_candidates[i];
      if (bare_bone_search ||
          ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(cand_id)))) {
        if (!has_valid_candidates) {
          lowerBound = dist;
          has_valid_candidates = true;
        }
        if (!bare_bone_search && stop_condition) {
          stop_condition->add_point_to_result(
              getExternalLabel(cand_id), getDataByInternalId(cand_id), dist);
        }
      }
      candidate_set[candidate_set_size++] = std::make_pair(dist, cand_id);
      std::push_heap(candidate_set, candidate_set + candidate_set_size,
                     compare_by_first_rev);
    }

    while (candidate_set_size > 0) {
      std::pair<dist_t, tableint> current_node_pair = candidate_set[0];

      dist_t candidate_dist = current_node_pair.first;

      bool flag_stop_search;
      if (bare_bone_search) {
        flag_stop_search = candidate_dist > lowerBound;
      } else {
        if (stop_condition) {
          flag_stop_search =
              stop_condition->should_stop_search(candidate_dist, lowerBound);
        } else {
          flag_stop_search =
              candidate_dist > lowerBound && top_candidates_size == ef;
        }
      }
      if (flag_stop_search) {
        break;
      }
      std::pop_heap(candidate_set, candidate_set + candidate_set_size--,
                    compare_by_first_rev);

      tableint current_node_id = current_node_pair.second;

      char *element = elements_ + current_node_id * size_data_per_element_;
      char *neighbors = get_neighbors(element);
      if (neighbors == nullptr) {
        continue;
      }
      levelsizeint element_level = get_element_level(element);
      size_t size = element_level == 0 ? get_total_neighbor(element)
                                       : ((offsetint *)(neighbors))[0];
      if (size == 0)
        continue;
      tableint *data =
          (tableint *)(neighbors + sizeof(offsetint) * element_level);

      if (collect_metrics) {
        metric_hops++;
        metric_distance_computations += size;
      }

#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *data), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *data + 64), _MM_HINT_T0);
      _mm_prefetch((char *)getDataByInternalId(*data), _MM_HINT_T0);
      _mm_prefetch((char *)(data + 1), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        int candidate_id = data[j];
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
        if (j + 1 < size)
          _mm_prefetch(getDataByInternalId(*(data + j + 1)),
                       _MM_HINT_T0); ////////////
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          char *currObj1 = (getDataByInternalId(candidate_id));
          dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

          bool flag_consider_candidate;
          if (!bare_bone_search && stop_condition) {
            flag_consider_candidate =
                stop_condition->should_consider_candidate(dist, lowerBound);
          } else {
            flag_consider_candidate =
                top_candidates_size < ef || lowerBound > dist;
          }

          if (flag_consider_candidate) {
            candidate_set[candidate_set_size++] =
                std::make_pair(dist, candidate_id);
            std::push_heap(candidate_set, candidate_set + candidate_set_size,
                           compare_by_first_rev);
#ifdef USE_SSE
            _mm_prefetch(
                getDataByInternalId(candidate_set[0].second), ///////////
                _MM_HINT_T0); ////////////////////////
#endif

            if (bare_bone_search ||
                (!isMarkedDeleted(candidate_id) &&
                 ((!isIdAllowed) ||
                  (*isIdAllowed)(getExternalLabel(candidate_id))))) {
              top_candidates[top_candidates_size++] =
                  std::make_pair(dist, candidate_id);
              std::push_heap(top_candidates,
                             top_candidates + top_candidates_size,
                             compare_by_first);
              if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(
                    getExternalLabel(candidate_id), currObj1, dist);
              }
            }

            bool flag_remove_extra = false;
            if (!bare_bone_search && stop_condition) {
              flag_remove_extra = stop_condition->should_remove_extra();
            } else {
              flag_remove_extra = top_candidates_size > ef;
            }
            while (flag_remove_extra) {
              tableint id = top_candidates[0].second;
              std::pop_heap(top_candidates,
                            top_candidates + top_candidates_size--,
                            compare_by_first);
              if (!bare_bone_search && stop_condition) {
                stop_condition->remove_point_from_result(
                    getExternalLabel(id), getDataByInternalId(id), dist);
                flag_remove_extra = stop_condition->should_remove_extra();
              } else {
                flag_remove_extra = top_candidates_size > ef;
              }
            }

            if (top_candidates_size > 0)
              lowerBound = top_candidates[0].first;
          }
        }
      }
    }
    // free(candidate_set);
  }

  char *get_neighbors(tableint internal_id) const {
    return get_neighbors(elements_ + internal_id * size_data_per_element_);
  }

  char *get_neighbors(char *element) const {
    char **neighbors = (char **)(element + offsetNeighbor_);
    if (neighbors == nullptr) {
      return nullptr;
    }
    return *neighbors;
  }

  offsetint get_neighbor_offset_at_level(char *neighbors, int level) const {
    return level == 0 ? 0 : ((offsetint *)(neighbors))[level - 1];
  }

  levelsizeint get_element_level(char *element) const {
    return *(levelsizeint *)element;
  }

  levelsizeint get_total_neighbor(char *element) const {
    return *(linklistsizeint *)(element + offsetTotalNeighbor_);
  }

  size_t get_neighbor_size(tableint internal_id) const {
    char *element = elements_ + internal_id * size_data_per_element_;
    if (element == nullptr) {
      return 0;
    }
    size_t element_level = get_element_level(element);
    size_t total_neighbor = get_total_neighbor(element);
    return sizeof(offsetint) * element_level +
           sizeof(tableint) * total_neighbor;
  }

  size_t indexFileSize() const {
    size_t size = 0;
    size += sizeof(cur_element_count_);
    size += sizeof(size_data_per_element_);
    size += sizeof(label_offset_);
    size += sizeof(offsetTotalNeighbor_);
    size += sizeof(offsetData_);
    size += sizeof(offsetNeighbor_);
    size += sizeof(maxlevel_);
    size += sizeof(threshold_level_);
    size += sizeof(enterpoint_node_);
    size += sizeof(maxM_);

    size += sizeof(maxM0_);
    size += sizeof(M_);
    size += sizeof(ef_construction_);

    size += cur_element_count_ * size_data_per_element_;

    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int neighborsSize = get_neighbor_size(i);
      size += sizeof(neighborsSize);
      size += neighborsSize;
    }
    return size;
  }

  void saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, cur_element_count_);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetTotalNeighbor_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, offsetNeighbor_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, threshold_level_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, ef_construction_);

    writeBinaryPOD(output, has_deleted_elements_);

    output.write(elements_, cur_element_count_ * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int elementSize = get_neighbor_size(i);
      size_t total_neighbors = get_total_neighbor(
          elements_ + i * size_data_per_element_);
      writeBinaryPOD(output, elementSize);
      if (elementSize && total_neighbors != 0) {
        assert(get_neighbors(i) != nullptr);
        output.write(get_neighbors(i), elementSize);
      }
    }
    output.close();
  }

  void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                 size_t max_elements_i = 0) {
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open())
      throw std::runtime_error("Cannot open file");

    clear();
    // todo: set max_elements_
    max_elements_ = max_elements_i;
    readBinaryPOD(input, cur_element_count_);

    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetTotalNeighbor_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, offsetNeighbor_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, threshold_level_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, ef_construction_);

    readBinaryPOD(input, has_deleted_elements_);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    elements_ = (char *)malloc(max_elements_ * size_data_per_element_);
    if (elements_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate "
          "elements array");
    input.read(elements_, cur_element_count_ * size_data_per_element_);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements_));

    ef_ = 10;
    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int neighborsSize;
      readBinaryPOD(input, neighborsSize);
      char **neighbors_ptr =
          (char **)(elements_ + i * size_data_per_element_ + offsetNeighbor_);
      if (neighborsSize == 0 ||
          get_total_neighbor(elements_ + i * size_data_per_element_) == 0) {
        *neighbors_ptr = nullptr;
      } else {
        *neighbors_ptr = (char *)malloc(neighborsSize);
        if (*neighbors_ptr == nullptr)
          throw std::runtime_error(
              "Not enough memory: loadIndex failed to allocate linklist");
        input.read(*neighbors_ptr, neighborsSize);
      }
    }

    input.close();

    return;
  }

  void convertFromHNSW(HierarchicalNSW<dist_t> *hnsw) {
    // clear();
    cur_element_count_ = hnsw->cur_element_count;
    has_deleted_elements_ = hnsw->num_deleted_ > 0;

    maxM_ = hnsw->maxM_;
    maxM0_ = hnsw->maxM0_;
    M_ = hnsw->M_;
    ef_construction_ = hnsw->ef_construction_;
    ef_ = hnsw->ef_;

    offsetTotalNeighbor_ = sizeof(levelsizeint);
    label_offset_ = offsetTotalNeighbor_ + sizeof(linklistsizeint);
    offsetNeighbor_ = label_offset_ + sizeof(labeltype);
    offsetData_ = offsetNeighbor_ + sizeof(char **);
    size_data_per_element_ = offsetData_ + data_size_;

    maxlevel_ = hnsw->maxlevel_;
    enterpoint_node_ = hnsw->enterpoint_node_;

    data_size_ = hnsw->data_size_;
    fstdistfunc_ = hnsw->fstdistfunc_;
    dist_func_param_ = hnsw->dist_func_param_;

    label_lookup_ = hnsw->label_lookup_;

    if (hnsw->max_elements_ > max_elements_ || elements_ == nullptr) {
      max_elements_ = hnsw->max_elements_;
      elements_ = (char *)malloc(max_elements_ * size_data_per_element_);
      if (elements_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: convertFromHNSW failed to allocate elements "
            "array");

      visited_list_pool_ = std::unique_ptr<VisitedListPool>(
          new VisitedListPool(1, max_elements_));
    }

    std::vector<std::vector<size_t>> degree_histogram(
        maxlevel_ + 1, std::vector<size_t>(maxM0_ + 2, 0)); // [0, maxM0_+1]
    std::vector<size_t> level_cnts(maxlevel_ + 1, 0);
#pragma omp parallel for schedule(dynamic)
    for (tableint i = 0; i < cur_element_count_; i++) {
      levelsizeint element_level = hnsw->element_levels_[i];
      for (int l = 1; l <= element_level; l++) {
#pragma omp atomic
        level_cnts[l]++;
        linklistsizeint *ll_cur = hnsw->get_linklist(i, l);
        int size = hnsw->getListCount(ll_cur);
#pragma omp atomic
        degree_histogram[l][size]++;
      }
      linklistsizeint *ll_cur = hnsw->get_linklist0(i);
      int size = hnsw->getListCount(ll_cur);
#pragma omp atomic
      degree_histogram[0][size]++;
    }

    size_t acc = 0;
    std::vector<size_t> degree_threshold(maxlevel_ + 1, 0);
    size_t topN =
        static_cast<size_t>(level_cnts[0] * top_degree_percent0_ + 0.5);
    for (size_t d = degree_histogram[0].size() - 1; d > 0; --d) {
      acc += degree_histogram[0][d];
      if (acc >= topN) {
        degree_threshold[0] = d;
        break;
      }
    }
    for (size_t l = 1; l <= maxlevel_; l++) {
      acc = 0;
      size_t topN =
          static_cast<size_t>(level_cnts[l] * top_degree_percent_ + 0.5);
      for (size_t d = degree_histogram[l].size() - 1; d > 0; --d) {
        acc += degree_histogram[l][d];
        if (acc >= topN) {
          degree_threshold[l] = d;
          break;
        }
      }
    }


    std::vector<std::vector<std::vector<tableint>>> new_neighbors_by_level(
        maxlevel_ + 1, std::vector<std::vector<tableint>>(cur_element_count_));
#pragma omp parallel for schedule(dynamic)
    for (tableint v = 0; v < cur_element_count_; v++) {
      thread_local std::priority_queue<
          std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
          typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst>
          heap;
      thread_local tableint *nbrs =
          (tableint *)malloc(sizeof(tableint) * maxM0_);

      levelsizeint v_level = hnsw->element_levels_[v];

      for (int l = 0; l <= v_level; l++) {
        size_t nbrs_size = 0;
        linklistsizeint *ll_cur = hnsw->get_linklist_at_level(v, l);
        int size = hnsw->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);

        size_t M0;
        if (l == 0) {
          M0 = size > degree_threshold[l] ? top_degree_M0_ : low_degree_m0_;
        } else {
          M0 = size > degree_threshold[l] ? top_degree_M_ : low_degree_m_;
        }

        std::vector<dist_t> dists(size);
#pragma omp parallel for
        for (int j = 0; j < size; j++) {
          tableint neighbor_id = data[j];
          dists[j] = hnsw->fstdistfunc_(hnsw->getDataByInternalId(v),
                                        hnsw->getDataByInternalId(neighbor_id),
                                        hnsw->dist_func_param_);
        }
        for (int j = 0; j < size; j++) {
          heap.emplace(dists[j], data[j]);
        }

        hnsw->getNeighborsByHeuristic2(heap, M0);
        while (!heap.empty()) {
          nbrs[nbrs_size++] = heap.top().second;
          heap.pop();
        }
        new_neighbors_by_level[l][v].insert(
            new_neighbors_by_level[l][v].begin(), nbrs, nbrs + nbrs_size);
      }
    }

#pragma omp parallel for schedule(dynamic)
    for (int l = 0; l <= maxlevel_; l++) {
      for (tableint v = 0; v < cur_element_count_; v++) {
        for (tableint u : new_neighbors_by_level[l][v]) {
          new_neighbors_by_level[l][u].push_back(v);
        }
      }
    }

#pragma omp parallel for schedule(dynamic)
    for (tableint i = 0; i < cur_element_count_; i++) {

      size_t debug_size = sizeof(tableint) * (maxlevel_ * maxM_ + maxM0_);

      thread_local tableint *neighbors =
          (tableint *)malloc(sizeof(tableint) * (maxlevel_ * maxM_ + maxM0_));
      thread_local offsetint *offsets =
          (offsetint *)malloc(sizeof(offsetint) * (maxlevel_ + 2));

      char *element = elements_ + i * size_data_per_element_;
      levelsizeint element_level = hnsw->element_levels_[i];
      memcpy(element, &element_level, sizeof(levelsizeint));
      memcpy(element + offsetData_, hnsw->getDataByInternalId(i), data_size_);
      memcpy(element + label_offset_, hnsw->getExternalLabeLp(i),
             sizeof(labeltype));

      size_t total_neighbor = 0;
      size_t offset_size = 0;

      for (int l = 0; l <= element_level; l++) {
        auto &nbrs = new_neighbors_by_level[l][i];
        std::sort(nbrs.begin(), nbrs.end());
        auto unique_end = std::unique(nbrs.begin(), nbrs.end());

        size_t size = unique_end - nbrs.begin();
        size_t limit = (l == 0) ? maxM0_ : maxM_;
        bool to_free = false;
        thread_local tableint *nbrs_prune = nullptr;
        if (unique_end - nbrs.begin() > limit) {
          std::priority_queue<
              std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
              typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst>
              heap;

          std::vector<dist_t> dists(size);
#pragma omp parallel for
          for (int j = 0; j < size; j++) {
            tableint neighbor_id = nbrs[j];
            dists[j] = hnsw->fstdistfunc_(hnsw->getDataByInternalId(i),
                                          hnsw->getDataByInternalId(neighbor_id),
                                          hnsw->dist_func_param_);
          }
          for (int j = 0; j < size; j++) {
            heap.emplace(dists[j], nbrs[j]);
          }

          hnsw->getNeighborsByHeuristic2(heap, limit);
          nbrs_prune = (tableint*)malloc(sizeof(tableint) * limit);
          to_free = true;
          for (size_t j=0; j<limit; j++) {
            nbrs_prune[j] = heap.top().second;
            heap.pop();
          }
          size = limit;

        } else {
          nbrs_prune = nbrs.data();
        }

        if (l == threshold_level_) {
          if ((total_neighbor + size) * sizeof(tableint) > debug_size) {
            std::cout << "error_size: 998" << std::endl;
          }

          memcpy(neighbors + total_neighbor, nbrs_prune, size * sizeof(tableint));
          total_neighbor += size;
          if (total_neighbor >= std::numeric_limits<linklistsizeint>::max()) {
            throw std::runtime_error(
                "Neighbor size exceeds the maximum allowed size");
          }
        } else {
          for (size_t j = 0;  j < size; j++) {
            tableint neighbor_id = nbrs_prune[j];
            assert(neighbor_id < cur_element_count_);
            if (hnsw->element_levels_[neighbor_id] == l) {
              neighbors[total_neighbor] = neighbor_id;
              if (total_neighbor++ >=
                  std::numeric_limits<linklistsizeint>::max()) {
                throw std::runtime_error(
                    "Neighbor size exceeds the maximum allowed size");
                  }
            }
          }
        }
        if (to_free) {
          free(nbrs_prune);
        }
        offsets[offset_size++] = total_neighbor;
      }

      memcpy(element + offsetTotalNeighbor_, &total_neighbor,
             sizeof(levelsizeint));

      if (total_neighbor == 0) {
        *(char **)(element + offsetNeighbor_) = nullptr;
        continue;
      }

      char *neighbors_ptr = (char *)malloc(sizeof(offsetint) * element_level +
                                           sizeof(tableint) * total_neighbor);
      if (neighbors_ptr == nullptr)
        throw std::runtime_error("Not enough memory: convertFromHNSW "
                                 "failed to allocate linklist");
      *(char **)(element + offsetNeighbor_) = neighbors_ptr;

      __builtin_memcpy(neighbors_ptr, offsets,
                       sizeof(offsetint) * element_level);
      __builtin_memcpy(neighbors_ptr + sizeof(offsetint) * element_level,
                       neighbors, sizeof(tableint) * total_neighbor);
    }
  }

  void convertFromHNSWWithDiff(HierarchicalNSW<dist_t> *hnsw,
                               std::ostream &output) {
    size_t prev_element_count = cur_element_count_;
    cur_element_count_ = hnsw->cur_element_count;
    has_deleted_elements_ = hnsw->num_deleted_ > 0;
    maxlevel_ = hnsw->maxlevel_;
    enterpoint_node_ = hnsw->enterpoint_node_;
    data_size_ = hnsw->data_size_;

    std::cout << "lookup" << std::endl;
    label_lookup_.merge(hnsw->label_lookup_);
    std::cout << "lookup-end" << std::endl;


    if (hnsw->max_elements_ > max_elements_ || elements_ == nullptr) {
      max_elements_ = hnsw->max_elements_;
      elements_ =
          (char *)realloc(elements_, max_elements_ * size_data_per_element_);
      if (elements_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: convertFromHNSW failed to allocate elements "
            "array");

      visited_list_pool_ = std::unique_ptr<VisitedListPool>(
          new VisitedListPool(1, max_elements_));
    }

    std::atomic<size_t> changed_old_nodes_count{0};
    std::atomic<size_t> changed_new_nodes_count{0};
    folly::atomic_grow_array<tableint> changed_old_nodes;
    folly::atomic_grow_array<tableint> changed_new_nodes;

    std::vector<std::vector<size_t>> degree_histogram(
        maxlevel_ + 1, std::vector<size_t>(maxM0_ + 2, 0)); // [0, maxM0_+1]
    std::vector<size_t> level_cnts(maxlevel_ + 1, 0);
#pragma omp parallel for schedule(dynamic)
    for (tableint i = 0; i < cur_element_count_; i++) {
      levelsizeint element_level = hnsw->element_levels_[i];
      for (int l = 1; l <= element_level; l++) {
#pragma omp atomic
        level_cnts[l]++;
        linklistsizeint *ll_cur = hnsw->get_linklist(i, l);
        int size = hnsw->getListCount(ll_cur);
#pragma omp atomic
        degree_histogram[l][size]++;
      }
      linklistsizeint *ll_cur = hnsw->get_linklist0(i);
      int size = hnsw->getListCount(ll_cur);
#pragma omp atomic
      degree_histogram[0][size]++;
    }

    size_t acc = 0;
    std::vector<size_t> degree_threshold(maxlevel_ + 1, 0);
    size_t topN =
        static_cast<size_t>(level_cnts[0] * top_degree_percent0_ + 0.5);
    for (size_t d = degree_histogram[0].size() - 1; d > 0; --d) {
      acc += degree_histogram[0][d];
      if (acc >= topN) {
        degree_threshold[0] = d;
        break;
      }
    }
    for (size_t l = 1; l <= maxlevel_; l++) {
      acc = 0;
      size_t topN =
          static_cast<size_t>(level_cnts[l] * top_degree_percent_ + 0.5);
      for (size_t d = degree_histogram[l].size() - 1; d > 0; --d) {
        acc += degree_histogram[l][d];
        if (acc >= topN) {
          degree_threshold[l] = d;
          break;
        }
      }
    }

    std::vector<std::vector<std::vector<tableint>>> new_neighbors_by_level(
        maxlevel_ + 1, std::vector<std::vector<tableint>>(cur_element_count_));
#pragma omp parallel for schedule(dynamic)
    for (tableint v = 0; v < cur_element_count_; v++) {
      thread_local std::priority_queue<
          std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
          typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst>
          heap;
      thread_local tableint *nbrs =
          (tableint *)malloc(sizeof(tableint) * maxM0_);
      levelsizeint v_level = hnsw->element_levels_[v];

      for (int l = v_level; l >= 0; l--) {
        size_t nbrs_size = 0;
        linklistsizeint *ll_cur = hnsw->get_linklist_at_level(v, l);
        int size = hnsw->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);

        size_t M0;
        if (l == 0) {
          M0 = size > degree_threshold[l] ? top_degree_M0_ : low_degree_m0_;
        } else {
          M0 = size > degree_threshold[l] ? top_degree_M_ : low_degree_m_;
        }

        std::vector<dist_t> dists(size);
#pragma omp parallel for
        for (int j = 0; j < size; j++) {
          tableint neighbor_id = data[j];
          dists[j] = hnsw->fstdistfunc_(hnsw->getDataByInternalId(v),
                                        hnsw->getDataByInternalId(neighbor_id),
                                        hnsw->dist_func_param_);
        }
        for (int j = 0; j < size; j++) {
          heap.emplace(dists[j], data[j]);
        }

        hnsw->getNeighborsByHeuristic2(heap, M0);
        while (!heap.empty()) {
          nbrs[nbrs_size++] = heap.top().second;
          assert(heap.top().second < cur_element_count_);
          heap.pop();
        }
        new_neighbors_by_level[l][v].insert(
            new_neighbors_by_level[l][v].begin(), nbrs, nbrs + nbrs_size);
      }
    }

#pragma omp parallel for schedule(dynamic)
      for (int l = 0; l <= maxlevel_; l++) {
        for (tableint v = 0; v < cur_element_count_; v++) {
          for (tableint u : new_neighbors_by_level[l][v]) {
            new_neighbors_by_level[l][u].push_back(v);
          }
        }
      }

#pragma omp parallel for schedule(dynamic)
    for (tableint i = 0; i < cur_element_count_; i++) {
      thread_local tableint *neighbors =
          (tableint *)malloc(sizeof(tableint) * (maxlevel_ * maxM_ + maxM0_));
      thread_local offsetint *offsets =
          (offsetint *)malloc(sizeof(offsetint) * (maxlevel_ + 2));

      char *element = elements_ + i * size_data_per_element_;
      char *prev_neighbors_ptr = get_neighbors(element);
      size_t prev_neighbors_size =
          sizeof(offsetint) * get_element_level(element) +
          sizeof(tableint) * get_total_neighbor(element);
      levelsizeint element_level = hnsw->element_levels_[i];

      memcpy(element, &element_level, sizeof(levelsizeint));
      memcpy(element + offsetData_, hnsw->getDataByInternalId(i), data_size_);
      memcpy(element + label_offset_, hnsw->getExternalLabeLp(i),
             sizeof(labeltype));

      size_t total_neighbor = 0;
      size_t offset_size = 0;
      for (int l = 0; l <= element_level; l++) {
        auto &nbrs = new_neighbors_by_level[l][i];
        for (auto tmps : new_neighbors_by_level[l][i]) {
          assert(tmps < cur_element_count_);
        }
        std::sort(nbrs.begin(), nbrs.end());
        for (auto tmps : new_neighbors_by_level[l][i]) {
          assert(tmps < cur_element_count_);
        }
        auto unique_end = std::unique(nbrs.begin(), nbrs.end());

        size_t size = unique_end - nbrs.begin();
        size_t limit = (l == 0) ? maxM0_ : maxM_;
        bool to_free = false;
        thread_local tableint *nbrs_prune = nullptr;
        if (unique_end - nbrs.begin() > limit) {
          std::priority_queue<
              std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
              typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst>
              heap;
          std::vector<dist_t> dists(size);
#pragma omp parallel for
          for (int j = 0; j < size; j++) {
            tableint neighbor_id = nbrs[j];
            dists[j] = hnsw->fstdistfunc_(hnsw->getDataByInternalId(i),
                                          hnsw->getDataByInternalId(neighbor_id),
                                          hnsw->dist_func_param_);
          }
          for (int j = 0; j < size; j++) {
            heap.emplace(dists[j], nbrs[j]);
          }
          hnsw->getNeighborsByHeuristic2(heap, limit);
          nbrs_prune = (tableint*)malloc(sizeof(tableint) * limit);
          to_free = true;
          for (size_t j=0; j<limit; j++) {
            nbrs_prune[j] = heap.top().second;
            heap.pop();
          }
          // std::sort(nbrs_prune, nbrs_prune + limit);
          size = limit;

        } else {
          nbrs_prune = nbrs.data();
        }

        if (l == threshold_level_) {
          memcpy(neighbors + total_neighbor, nbrs_prune, size * sizeof(tableint));
          total_neighbor += size;
          if (total_neighbor >= std::numeric_limits<linklistsizeint>::max()) {
            throw std::runtime_error(
                "Neighbor size exceeds the maximum allowed size");
          }
        } else {
          for (size_t j = 0;  j < size; j++) {
            tableint neighbor_id = nbrs_prune[j];
            assert(neighbor_id < cur_element_count_);
            if (hnsw->element_levels_[neighbor_id] == l) {
              neighbors[total_neighbor] = neighbor_id;
              if (total_neighbor++ >=
                  std::numeric_limits<linklistsizeint>::max()) {
                throw std::runtime_error(
                    "Neighbor size exceeds the maximum allowed size");
                  }
            }
          }
        }
        if (to_free) {
          free(nbrs_prune);
        }

        offsets[offset_size++] = total_neighbor;
      }
      memcpy(element + offsetTotalNeighbor_, &total_neighbor,
             sizeof(levelsizeint));

      if (total_neighbor == 0) {
        *(char **)(element + offsetNeighbor_) = nullptr;
        continue;
      }

      size_t neighbors_size =
          sizeof(offsetint) * element_level + sizeof(tableint) * total_neighbor;


      char *neighbors_ptr = (char *)malloc(neighbors_size);
      if (neighbors_ptr == nullptr)
        throw std::runtime_error("Not enough memory: convertFromHNSW "
                                 "failed to allocate linklist");

      __builtin_memcpy(neighbors_ptr, offsets,
                       sizeof(offsetint) * element_level);
      __builtin_memcpy(neighbors_ptr + sizeof(offsetint) * element_level,
                       neighbors, sizeof(tableint) * total_neighbor);

      *(char **)(element + offsetNeighbor_) = neighbors_ptr;

      // 判断是否新节点或邻居有变化
      if (prev_neighbors_size == 0 || prev_neighbors_ptr == nullptr) {
        if (i >= prev_element_count) {
          changed_new_nodes[changed_new_nodes_count++] = i;
        } else {
          changed_old_nodes[changed_old_nodes_count++] = i;
        }
      } else if (prev_neighbors_size != neighbors_size ||
                 memcmp(prev_neighbors_ptr, neighbors_ptr, neighbors_size) !=
                     0) {
        if (i >= prev_element_count) {
          changed_new_nodes[changed_new_nodes_count++] = i;
        } else {
          changed_old_nodes[changed_old_nodes_count++] = i;
        }
        free(prev_neighbors_ptr);
      }
    }


    std::cout << "changed nodes count: " << changed_old_nodes_count << " + "
              << changed_new_nodes_count << std::endl;

    writeBinaryPOD(output, cur_element_count_);

    size_t changed_old_cnt = changed_old_nodes_count;
    size_t changed_new_cnt = changed_new_nodes_count;
    writeBinaryPOD(output, changed_old_cnt);
    writeBinaryPOD(output, changed_new_cnt);
    for (size_t i = 0; i < changed_old_cnt; i++) {
      tableint v = changed_old_nodes[i];
      writeBinaryPOD(output, v);
      char *element = elements_ + v * size_data_per_element_;
      output.write(element, offsetNeighbor_);
      unsigned int neighborsSize = get_neighbor_size(v);
      writeBinaryPOD(output, neighborsSize);
      if (neighborsSize) {
        char *neighbors = get_neighbors(v);
        output.write(neighbors, neighborsSize);
      }
    }
    for (size_t i = 0; i < changed_new_cnt; i++) {
      tableint v = changed_new_nodes[i];
      writeBinaryPOD(output, v);
      char *element = elements_ + v * size_data_per_element_;
      output.write(element, size_data_per_element_);
      unsigned int neighborsSize = get_neighbor_size(v);
      writeBinaryPOD(output, neighborsSize);
      if (neighborsSize) {
        char *neighbors = get_neighbors(v);
        output.write(neighbors, neighborsSize);
      }
    }
  }

  template <typename data_t>
  std::vector<data_t> getDataByLabel(labeltype label) const {
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
      throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;

    char *data_ptrv = getDataByInternalId(internalId);
    size_t dim = *((size_t *)dist_func_param_);
    std::vector<data_t> data;
    data_t *data_ptr = (data_t *)data_ptrv;
    for (size_t i = 0; i < dim; i++) {
      data.push_back(*data_ptr);
      data_ptr += 1;
    }
    return data;
  }

  /*
   * Checks the first 16 bits of the memory to see if the element is marked
   * deleted.
   */
  bool isMarkedDeleted(tableint internalId) const {
    unsigned char *ll_cur = (unsigned char *)elements_ +
                            internalId * size_data_per_element_ +
                            sizeof(levelsizeint) + 2;
    return *ll_cur & DELETE_MARK;
  }

  std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void *query_data, size_t k,
            BaseFilterFunctor *isIdAllowed) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0)
      return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    visited_array[currObj] = visited_array_tag;
    for (int level = maxlevel_; level > threshold_level_; level--) {
      bool changed = true;
      while (changed) {
        changed = false;

        char *element = elements_ + currObj * size_data_per_element_;
        char *neighbors = get_neighbors(element);
        if (neighbors == nullptr) {
          continue;
        }
        levelsizeint element_level = get_element_level(element);
        offsetint offset = get_neighbor_offset_at_level(neighbors, level);
        size_t size =
            (level == element_level ? get_total_neighbor(element)
                                    : ((offsetint *)(neighbors))[level]) -
            offset;
        if (size == 0)
          continue;

        tableint *data =
            (tableint *)(neighbors + sizeof(offsetint) * element_level) +
            offset;

        metric_hops++;
        metric_distance_computations += size;

        for (size_t i = 0; i < size; i++) {
          tableint cand = data[i];
          if (cand < 0 || cand > cur_element_count_)
            throw std::runtime_error("cand error");
          dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                  dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    size_t ef = std::max(ef_, k);

    thread_local std::pair<dist_t, tableint> *top_candidates = nullptr;
    thread_local size_t top_candidates_capacity = 0;
    size_t need_size = ef + 1;
    if (top_candidates_capacity < need_size) {
      top_candidates = (std::pair<dist_t, tableint> *)realloc(
          top_candidates, need_size * sizeof(std::pair<dist_t, tableint>));
      top_candidates_capacity = need_size;
    }
    thread_local std::pair<dist_t, tableint> *candidate_set = nullptr;
    thread_local size_t candidate_set_capacity = 0;
    size_t max_candidate_set_size = ef * ef;
    if (candidate_set_capacity < max_candidate_set_size) {
      candidate_set = (std::pair<dist_t, tableint> *)realloc(
          candidate_set,
          max_candidate_set_size * sizeof(std::pair<dist_t, tableint>));
      candidate_set_capacity = max_candidate_set_size;
    }
    size_t top_candidates_size = 1;
    top_candidates[0] = std::make_pair(curdist, currObj);
    visited_array[currObj] = visited_array_tag;

    dist_t lowerBound = !isMarkedDeleted(currObj)
                            ? curdist
                            : std::numeric_limits<dist_t>::max();
    for (int level = std::min(threshold_level_, maxlevel_); level > 0;
         level--) {
      searchBaseLayer(query_data, level, top_candidates, top_candidates_size,
                      candidate_set, ef, visited_array, visited_array_tag,
                      lowerBound);
    }

    bool bare_bone_search = !has_deleted_elements_ && !isIdAllowed;
    if (bare_bone_search) {
      searchBaseLayerST<true>(query_data, top_candidates, top_candidates_size,
                              candidate_set, ef, visited_array,
                              visited_array_tag, isIdAllowed);
    } else {
      searchBaseLayerST<false>(query_data, top_candidates, top_candidates_size,
                               candidate_set, ef, visited_array,
                               visited_array_tag, isIdAllowed);
    }
    visited_list_pool_->releaseVisitedList(vl);

    while (top_candidates_size > k) {
      std::pop_heap(top_candidates, top_candidates + top_candidates_size--,
                    compare_by_first);
    }
    for (size_t i = 0; i < top_candidates_size; i++) {
      std::pair<dist_t, tableint> rez = top_candidates[i];
      result.emplace(rez.first, getExternalLabel(rez.second));
    }
    return result;
  }

  std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void *query_data, size_t k) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0)
      return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    visited_array[currObj] = visited_array_tag;
    for (int level = maxlevel_; level > threshold_level_; level--) {
      bool changed = true;
      while (changed) {
        changed = false;

        char *element = elements_ + currObj * size_data_per_element_;
        char *neighbors = get_neighbors(element);
        if (neighbors == nullptr) {
          continue;
        }
        levelsizeint element_level = get_element_level(element);
        offsetint offset = get_neighbor_offset_at_level(neighbors, level);
        size_t size =
            (level == element_level ? get_total_neighbor(element)
                                    : ((offsetint *)(neighbors))[level]) -
            offset;
        if (size == 0)
          continue;

        tableint *data =
            (tableint *)(neighbors + sizeof(offsetint) * element_level) +
            offset;

        metric_hops++;
        metric_distance_computations += size;

        for (size_t i = 0; i < size; i++) {
          tableint cand = data[i];
          if (cand < 0 || cand > cur_element_count_)
            throw std::runtime_error("cand error");
          dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                  dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    size_t ef = std::max(ef_, k);

    thread_local std::pair<dist_t, tableint> *top_candidates = nullptr;
    thread_local size_t top_candidates_capacity = 0;
    size_t need_size = ef + 1;
    if (top_candidates_capacity < need_size) {
      top_candidates = (std::pair<dist_t, tableint> *)realloc(
          top_candidates, need_size * sizeof(std::pair<dist_t, tableint>));
      top_candidates_capacity = need_size;
    }

    thread_local std::pair<dist_t, tableint> *candidate_set = nullptr;
    thread_local size_t candidate_set_capacity = 0;
    size_t max_candidate_set_size = ef * ef;
    if (candidate_set_capacity < max_candidate_set_size) {
      candidate_set = (std::pair<dist_t, tableint> *)realloc(
          candidate_set,
          max_candidate_set_size * sizeof(std::pair<dist_t, tableint>));
      candidate_set_capacity = max_candidate_set_size;
    }
    size_t top_candidates_size = 1;
    top_candidates[0] = std::make_pair(curdist, currObj);
    visited_array[currObj] = visited_array_tag;

    dist_t lowerBound = !isMarkedDeleted(currObj)
                            ? curdist
                            : std::numeric_limits<dist_t>::max();
    for (int level = std::min(threshold_level_, maxlevel_); level > 0;
         level--) {
      searchBaseLayer(query_data, level, top_candidates, top_candidates_size,
                      candidate_set, ef, visited_array, visited_array_tag,
                      lowerBound);
    }
    bool bare_bone_search = !has_deleted_elements_;
    if (bare_bone_search) {
      searchBaseLayerST<true>(query_data, top_candidates, top_candidates_size,
                              candidate_set, ef, visited_array,
                              visited_array_tag, lowerBound);
    } else {
      searchBaseLayerST<false>(query_data, top_candidates, top_candidates_size,
                               candidate_set, ef, visited_array,
                               visited_array_tag, lowerBound);
    }
    visited_list_pool_->releaseVisitedList(vl);

    while (top_candidates_size > k) {
      std::pop_heap(top_candidates, top_candidates + top_candidates_size--,
                    compare_by_first);
    }
    for (size_t i = 0; i < top_candidates_size; i++) {
      std::pair<dist_t, tableint> rez = top_candidates[i];
      result.emplace(rez.first, getExternalLabel(rez.second));
    }
    return result;
  }

  void searchKnn(const void *query_data, size_t k, tableint *result) const {
    if (cur_element_count_ == 0)
      return;
    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (int level = maxlevel_; level > threshold_level_; level--) {
      bool changed = true;
      while (changed) {
        changed = false;

        char *element = elements_ + currObj * size_data_per_element_;
        char *neighbors = get_neighbors(element);
        if (neighbors == nullptr) {
          continue;
        }
        levelsizeint element_level = get_element_level(element);
        offsetint offset = get_neighbor_offset_at_level(neighbors, level);
        size_t size =
            (level == element_level ? get_total_neighbor(element)
                                    : ((offsetint *)(neighbors))[level]) -
            offset;

        if (size == 0)
          continue;

        tableint *data =
            (tableint *)(neighbors + sizeof(offsetint) * element_level) +
            offset;

        metric_hops++;
        metric_distance_computations += size;

        for (size_t i = 0; i < size; i++) {
          tableint cand = data[i];
          dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                  dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    size_t ef = std::max(ef_, k);

    thread_local std::pair<dist_t, tableint> *top_candidates = nullptr;
    thread_local size_t top_candidates_capacity = 0;
    size_t need_size = ef + 1;
    if (top_candidates_capacity < need_size) {
      top_candidates = (std::pair<dist_t, tableint> *)realloc(
          top_candidates, need_size * sizeof(std::pair<dist_t, tableint>));
      top_candidates_capacity = need_size;
    }

    thread_local std::pair<dist_t, tableint> *candidate_set = nullptr;
    thread_local size_t candidate_set_capacity = 0;
    size_t max_candidate_set_size = ef * ef;
    if (candidate_set_capacity < max_candidate_set_size) {
      candidate_set = (std::pair<dist_t, tableint> *)realloc(
          candidate_set,
          max_candidate_set_size * sizeof(std::pair<dist_t, tableint>));
      candidate_set_capacity = max_candidate_set_size;
    }
    size_t top_candidates_size = 1;
    top_candidates[0] = std::make_pair(curdist, currObj);
    visited_array[currObj] = visited_array_tag;

    dist_t lowerBound = !isMarkedDeleted(currObj)
                            ? curdist
                            : std::numeric_limits<dist_t>::max();

    for (int level = std::min(threshold_level_, maxlevel_); level > 0;
         level--) {
      searchBaseLayer(query_data, level, top_candidates, top_candidates_size,
                      candidate_set, ef, visited_array, visited_array_tag,
                      lowerBound);
    }
    bool bare_bone_search = !has_deleted_elements_;
    if (bare_bone_search) {
      searchBaseLayerST<true>(query_data, top_candidates, top_candidates_size,
                              candidate_set, ef, visited_array,
                              visited_array_tag, lowerBound);
    } else {
      searchBaseLayerST<false>(query_data, top_candidates, top_candidates_size,
                               candidate_set, ef, visited_array,
                               visited_array_tag, lowerBound);
    }
    visited_list_pool_->releaseVisitedList(vl);

    std::nth_element(top_candidates, top_candidates + k,
                     top_candidates + top_candidates_size, compare_by_first);
    for (size_t i = 0; i < k; i++) {
      result[i] = getExternalLabel(top_candidates[i].second);
    }
  }

  std::vector<std::pair<dist_t, labeltype>>
  searchStopConditionClosest(const void *query_data,
                             BaseSearchStopCondition<dist_t> &stop_condition,
                             BaseFilterFunctor *isIdAllowed = nullptr) const {
    std::vector<std::pair<dist_t, labeltype>> result;
    std::vector<std::pair<dist_t, tableint>> top_candidates;
    if (cur_element_count_ == 0)
      return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        char *element = elements_ + currObj * size_data_per_element_;
        char *neighbors = get_neighbors(element);
        if (neighbors == nullptr) {
          continue;
        }
        offsetint offset = get_neighbor_offset_at_level(neighbors, level);
        size_t limit = get_total_neighbor(element) - offset;
        levelsizeint element_level = get_element_level(element);

        tableint *data =
            (tableint *)(neighbors + sizeof(offsetint) * element_level) +
            offset;

        size_t size = std::min(maxM_, limit);

        metric_hops++;
        metric_distance_computations += size;

        for (int i = 0; i < size; i++) {
          tableint cand = data[i];
          if (cand < 0 || cand > cur_element_count_)
            throw std::runtime_error("cand error");
          dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                  dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    top_candidates.reserve(ef_);
    top_candidates.emplace_back(curdist, currObj);
    top_candidates = searchBaseLayerST<false>(query_data, top_candidates, 0,
                                              visited_array, visited_array_tag,
                                              isIdAllowed, &stop_condition);

    visited_list_pool_->releaseVisitedList(vl);
    size_t sz = top_candidates.size();
    result.resize(sz);
    while (!top_candidates.empty()) {
      result[--sz] = top_candidates.top();
      top_candidates.pop();
    }

    stop_condition.filter_results(result);

    return result;
  }

  void patchFromStream(std::istream &in,
                       std::unordered_map<uint32_t, std::vector<float>> &new_data) {
    size_t prev_id_limit = cur_element_count_;
    readBinaryPOD(in, cur_element_count_);

    for (const auto &item : new_data) {
      uint32_t id = item.first;
      char *element = elements_ + id * size_data_per_element_;

      memcpy(element + offsetData_, item.second.data(), data_size_);
      labeltype label_id = id;
      memcpy(element + label_offset_, &label_id, sizeof(labeltype));
      label_lookup_[label_id] = id;
    }

    size_t changed_old_cnt = 0, changed_new_cnt = 0;
    readBinaryPOD(in, changed_old_cnt);
    readBinaryPOD(in, changed_new_cnt);

    for (size_t i = 0; i < changed_old_cnt + changed_new_cnt; ++i) {
      tableint id;
      readBinaryPOD(in, id);
      char *element = elements_ + id * size_data_per_element_;
      char *prev_neighbors_ptr = nullptr;
      if (i < changed_old_cnt)
        prev_neighbors_ptr = *(char **)(element + offsetNeighbor_);
      unsigned int neighborsSize = 0;
      readBinaryPOD(in, neighborsSize);
      if (prev_neighbors_ptr) {
        free(prev_neighbors_ptr);
      }
      char *neighbors_ptr = nullptr;
      if (neighborsSize == 0) {
        neighbors_ptr = nullptr;
      } else {
        neighbors_ptr = (char *)malloc(neighborsSize);
        if (neighbors_ptr == nullptr)
          throw std::runtime_error(
              "Not enough memory: patchFromStream failed to allocate linklist");
        in.read(neighbors_ptr, neighborsSize);
      }
      *(char **)(element + offsetNeighbor_) = neighbors_ptr;
    }
  }

  size_t indexSize() const {
    size_t elements_size =
        cur_element_count_ * (sizeof(char **) + sizeof(linklistsizeint));
    size_t neighbors_size = 0;
    for (size_t i = 0; i < cur_element_count_; i++) {
      neighbors_size += get_neighbor_size(i);
    }

    return elements_size + neighbors_size;
  }
};
} // namespace hnswlib
