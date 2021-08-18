cmake_minimum_required(VERSION 3.10)

project(tests)

# Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(GCC_COVERAGE_COMPILE_FLAGS "-O3 -fopenmp -g")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_executable(rnn_test rnn_test.cpp rnn.cpp)
target_link_libraries(rnn_test LINK_PUBLIC Eigen3::Eigen)

add_executable(lstm_test lstm_test.cpp lstm.cpp weight_init.cpp)
target_link_libraries(lstm_test LINK_PUBLIC Eigen3::Eigen)
