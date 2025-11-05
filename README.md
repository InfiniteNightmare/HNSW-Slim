
# HNSW-Flash

This repository provides the official implementation for the paper **HNSW-Slim: A Light-Weight Vector Index for Approximate Nearest Neighbor Search**.

The code is built upon the [hnswlib](https://github.com/nmslib/hnswlib).
## Prerequisites

### Datasets

**File format:** All datasets used in this project are stored in `.fvecs` and `.ivecs` formats. Each file contains multiple vectors, where each line starts with an integer $d$ (the dimension), followed by $d$ numbers representing the vector data. The values are of type `float` in `.fvecs` files and `integer` in `.ivecs` files. For more details, see the [TEXMEX corpus documentation](http://corpus-texmex.irisa.fr/).

The main datasets evaluated in the paper are summarized below:

| Dataset | Dimension | Base Size | Query Size |
| ------- | --------- | --------- | ---------- |
| COHERE [(link)](https://huggingface.co/datasets/Cohere/wikipedia-22-12-es-embeddings) | 768 | 1,000,000 | 10,000 |
| GIST [(link)](http://corpus-texmex.irisa.fr/) | 960 | 1,000,000 | 1,000 |
| SIFT [(link)](http://corpus-texmex.irisa.fr/) | 128 | 6,000,000 | 10,000 |
| DEEP [(link)](http://sites.skoltech.ru/compvision/noimi/) | 96 | 9,000,000 | 10,000 |

Example directory structure for the `SIFT` dataset:

    HNSW_Slim
    ├─data
    │  ├─sift
    │  │  ├─sift_base.fvecs
    │  │  ├─sift_query.fvecs
    │  │  └─sift_groundtruth.ivecs
    │  ├─...


### Baselines

The following baselines are included for comparison:

| **Algorithm** | **Reference** |
|--------------|--------------|
| **HNSW [(link)](https://github.com/nmslib/hnswlib)** | Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs |
| **LEANN*** | LEANN: A Low-Storage Vector Index |
| **NSG [(link)](https://github.com/ZJULearning/nsg)** | Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph |
| **NSSG [(link)](https://github.com/ZJULearning/SSG)** | High Dimensional Similarity Search With Satellite System Graph: Efficiency, Scalability, and Unindexed Query Compatibility |
| **Vamana [(link)](https://github.com/microsoft/DiskANN)** | DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search On a Single Node |

> ***Note:*** Since LEANN does not provide open-source code, we implemented its pruning algorithm based on the paper's description, using HNSW as the foundation.

### Environment

- Ubuntu 24.04.2 LTS
- cmake 3.28.3
- g++ 13.3.0 with C++20 support
- tcmalloc
- folly
- openmp
- zlib
- gflags

## Build

To build the project, run:

```bash
mkdir build && cd build
cmake ..
make -j
```

## Usage

Ensure you are in the `build` directory before running the following commands.

### Run HNSW-Slim
**HNSW-Slim:**
```bash
./main --dataset=DATASET --solve_strategy=hnsw_slim --m=M --k=K --ef_construction=EF_CONSTRUCTION --ef_search=EF_SEARCH --branching_factor=BRANCHING_FACTOR --threshold_level=THRESHOLD_LEVEL --top_degree_percent0=TOP_DEGREE_PERCENT0 --top_M0=TOP_M0 --level_ratio=LEVEL_RATIO --Mm_ratio=M_RATIO
```

Where:
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `K`: Number of nearest neighbors to search for.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `EF_SEARCH`: Maximum number of candidates retained during the search phase.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph (i.e., $1/p$ in the paper).
- `THRESHOLD_LEVEL`: Layer at which only this and higher layers are subject to hierarchical pruning (i.e., $L_t$ in the paper).
- `TOP_DEGREE_PERCENT0`: Percentage of nodes classified as high degree (i.e., $\alpha_0\%=\alpha\%$ in the paper, default 0.02).
- `TOP_M0`: Number of neighbors retained for each high-degree node in the base layer (i.e., $M_{h_0}$, default 32).
- `LEVEL_RATIO`: Ratio of neighbors for high-degree nodes in high layers to those in the base layer (i.e., $M_{h} : M_{h_0} \times 100$, default 25).
- `M_RATIO`: Ratio of neighbors for low-degree to high-degree nodes in the base layer (i.e., $M_{h_0} : M_{l_0} \times 100$, default 50).

**HNSW-Slim with RaBitQ quantization:**
```bash
./main --dataset=DATASET --solve_strategy=hnsw-slimq --m=M --k=K --ef_construction=EF_CONSTRUCTION --ef_search=EF_SEARCH --branching_factor=BRANCHING_FACTOR --threshold_level=THRESHOLD_LEVEL --top_degree_percent0=TOP_DEGREE_PERCENT0 --top_M0=TOP_M0 --level_ratio=LEVEL_RATIO --Mm_ratio=M_RATIO
```
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `K`: Number of nearest neighbors to search for.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `EF_SEARCH`: Maximum number of candidates retained during the search phase.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph (i.e., $1/p$ in the paper).
- `THRESHOLD_LEVEL`: Layer at which only this and higher layers are subject to hierarchical pruning (i.e., $L_t$ in the paper).
- `TOP_DEGREE_PERCENT0`: Percentage of nodes classified as high degree (i.e., $\alpha_0\%=\alpha\%$ in the paper, default 0.02).
- `TOP_M0`: Number of neighbors retained for each high-degree node in the base layer (i.e., $M_{h_0}$, default 32).
- `LEVEL_RATIO`: Ratio of neighbors for high-degree nodes in high layers to those in the base layer (i.e., $M_{h} : M_{h_0} \times 100$, default 25).
- `M_RATIO`: Ratio of neighbors for low-degree to high-degree nodes in the base layer (i.e., $M_{h_0} : M_{l_0} \times 100$, default 50).

**HNSW-SlimZERO:**
```bash
./main --dataset=DATASET --solve_strategy=hnsw-slimzero --m=M --k=K --ef_construction=EF_CONSTRUCTION --ef_search=EF_SEARCH --branching_factor=BRANCHING_FACTOR --threshold_level=THRESHOLD_LEVEL --top_degree_percent0=TOP_DEGREE_PERCENT0 --top_M0=TOP_M0 --level_ratio=LEVEL_RATIO --Mm_ratio=M_RATIO --min_indegree0=MIN_INDEGREE0 --min_indegree=MIN_INDEGREE
```
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `K`: Number of nearest neighbors to search for.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `EF_SEARCH`: Maximum number of candidates retained during the search phase.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph (i.e., $1/p$ in the paper).
- `THRESHOLD_LEVEL`: Layer at which only this and higher layers are subject to hierarchical pruning (i.e., $L_t$ in the paper).
- `TOP_DEGREE_PERCENT0`: Percentage of nodes classified as high degree (i.e., $\alpha_0\%=\alpha\%$ in the paper, default 0.02).
- `TOP_M0`: Number of neighbors retained for each high-degree node in the base layer (i.e., $M_{h_0}$, default 32).
- `LEVEL_RATIO`: Ratio of neighbors for high-degree nodes in high layers to those in the base layer (i.e., $M_{h} : M_{h_0} \times 100$, default 25).
- `M_RATIO`: Ratio of neighbors for low-degree to high-degree nodes in the base layer (i.e., $M_{h_0} : M_{l_0} \times 100$, default 50).
- `MIN_INDEGREE0`: Max in-degree of retained neighbors of each node in the base layer;
- `MIN_INDEGREE`: Max in-degree of retained neighbors of each node in high layers;

### Update HNSW-Slim
**Generate HNSW-Slim index with partially loaded data** (HNSW_SLIMQ and HNSW_SLIMZERO not supported)
```bash
./main_partial --dataset=DATASET --solve_strategy=hnsw_slim --k=K --m=M --ef_construction=EF_CONSTRUCTION --branching_factor=BRANCHING_FACTOR --threshold_level=THREASHOLD_LEVEL --partial=PARTIAL
```
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph.
- `EF_SEARCH`: Maximum number of candidates retained during the search phase.
- `THRESHOLD_LEVEL`: Layer at which only this and higher layers are subject to hierarchical pruning (i.e., $L_t$ in the paper).
- `PARTIAL`: Ratio (%) of the partially loaded data.
- Other parameters use default values.

**Server:**
```bash
./hnsw_slim_server --dataset=DATASET --m=M --ef_construction=EF_CONSTRUCTION --branching_factor=BRANCHING_FACTOR --ef_search=EF_SEARCH --partial=PARTIAL
```
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph.
- `EF_SEARCH`: Maximum number of candidates retained during the search phase.
- `PARTIAL`: Ratio (%) of the partially loaded data.
- Other parameters use default values.

**Client:**
```bash
./hnsw_slim_client_update --dataset=DATASET --m=M --ef_construction=EF_CONSTRUCTION --branching_factor=BRANCHING_FACTOR --partial=PARTIAL --update_size=UPDATE_SIZE
```
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph.
- `PARTIAL`: Ratio (%) of the initially loaded data.
- `UPDATE_SIZE`: Size of data to be updated in each batch.
- Other parameters use default values.


**Server with deletion and reinsertion:**
```bash
./hnsw_slim_server_patch --dataset=DATASET --m=M --ef_construction=EF_CONSTRUCTION --branching_factor=BRANCHING_FACTOR --ef_search=EF_SEARCH --partial=PARTIAL --delete_rate=DELETE_RATE
```
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph.
- `EF_SEARCH`: Maximum number of candidates retained during the search phase.
- `PARTIAL`: Ratio (%) of the partially loaded data.
- `DELETE_RATE`: Ratio (%) of data partially deleted during the update phase.
- Other parameters use default values.

**Client with deletion and reinsertion:**
```bash
./hnsw_slim_client_update --dataset=DATASET --m=M --ef_construction=EF_CONSTRUCTION --branching_factor=BRANCHING_FACTOR --partial=PARTIAL --update_size=UPDATE_SIZE --k=K --ef_search=EF_SEARCH
```
- `DATASET`: Name of the dataset.
- `M`: Maximum number of outgoing connections in the HNSW graph.
- `EF_CONSTRUCTION`: Maximum number of candidate neighbors considered during index construction.
- `BRANCHING_FACTOR`: Branching factor for the HNSW graph.
- `PARTIAL`: Ratio (%) of the initially loaded data.
- `UPDATE_SIZE`: Size of data to be updated in each batch.
- `K`: Number of nearest neighbors to search for.
- `EF_SEARCH`: Maximum number of candidates retained during the search phase.
- Other parameters use default values.