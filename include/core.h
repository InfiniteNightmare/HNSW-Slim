#pragma once

#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

size_t K = 10;					// top-k knns
size_t M = 32; // neighbor's number, should be times of 16
size_t M0 = 32; // neighbor's number of level 0, should be times of 16
size_t EF_CONSTRUCTION = 1024;		// maximum number of candidate neighbors considered during index construction.
size_t EF_SEARCH = 64;			// maximum number of candidates retained during the search phase.

std::string BRANCHING_FACTOR = "4"; // branching factor of the HNSW graph

size_t THRESHOLD_LEVEL = 0; // threshold level for shrinking the HNSW graph