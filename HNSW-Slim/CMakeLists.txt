cmake_minimum_required(VERSION 3.15)
project(HYBRID)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -s -DNDEBUG -pthread -flto -march=native -ffat-lto-objects -w -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
set(CMAKE_CXX_FLAGS_DEBUG  "-pipe -O0 -fno-omit-frame-pointer -g -pthread -w -DINFO -march=native")

# Includes
set(HYBRID_SRC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(HYBRID_THIRD_PARTY_INCLUDE_DIR
        ${PROJECT_SOURCE_DIR}/third_party
)

include_directories(${HYBRID_SRC_INCLUDE_DIR} ${HYBRID_THIRD_PARTY_INCLUDE_DIR})

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP is not found")
endif()

# Protobuf
find_package(Protobuf REQUIRED)
if (Protobuf_FOUND)
    message(STATUS "Found Protobuf: ${Protobuf_VERSION}")
    include_directories(${Protobuf_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Protobuf not found!")
endif()

# folly
find_package(folly REQUIRED)
if (folly_FOUND)
    message(STATUS "Found folly: ${folly_VERSION}")
    include_directories(${folly_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "folly not found!")
endif()

# zlib
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "zlib not found!")
endif ()


protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROJECT_SOURCE_DIR}/query.proto)

add_executable(main main.cc)
add_executable(prep_data prep_data.cc)

add_executable(hnsw_server hnsw_server.cc ${PROTO_SRCS} ${PROTO_HDRS})

target_include_directories(hnsw_server PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)

target_link_libraries(main PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly)
target_link_libraries(prep_data PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly)

target_link_libraries(hnsw_server PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly ${Protobuf_LIBRARIES})

add_executable(main_partial main_partial.cc)
target_link_libraries(main_partial PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly)

add_executable(recall_knn recall_knn.cc)
target_link_libraries(recall_knn PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly)


add_executable(hnsw_slim_server hnsw_slim_server.cc ${PROTO_SRCS} ${PROTO_HDRS})
add_executable(hnsw_slim_client_update hnsw_slim_client_update.cc ${PROTO_SRCS} ${PROTO_HDRS})

target_include_directories(hnsw_slim_server PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(hnsw_slim_server PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly ${Protobuf_LIBRARIES})

target_include_directories(hnsw_slim_client_update PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(hnsw_slim_client_update PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly ${Protobuf_LIBRARIES})


target_link_libraries(hnsw_slim_server PRIVATE ${ZLIB_LIBRARIES})
target_link_libraries(hnsw_slim_client_update PRIVATE ${ZLIB_LIBRARIES})

add_executable(hnsw_slim_server_patch hnsw_slim_server_patch.cc ${PROTO_SRCS} ${PROTO_HDRS})
add_executable(hnsw_slim_client_update_patch hnsw_slim_client_update_patch.cc ${PROTO_SRCS} ${PROTO_HDRS})

target_include_directories(hnsw_slim_server_patch PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(hnsw_slim_server_patch PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly ${Protobuf_LIBRARIES})

target_include_directories(hnsw_slim_client_update_patch PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(hnsw_slim_client_update_patch PRIVATE stdc++fs gflags  glog tcmalloc_minimal folly ${Protobuf_LIBRARIES})


target_link_libraries(hnsw_slim_server_patch PRIVATE ${ZLIB_LIBRARIES})
target_link_libraries(hnsw_slim_client_update_patch PRIVATE ${ZLIB_LIBRARIES})

