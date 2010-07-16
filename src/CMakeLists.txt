cuda_add_library(twgame twgame_gpu.cu twgame_cpu.cpp)

cuda_add_executable(quick_check qcheck.cu)

add_executable(evolver evolver.cpp)
target_link_libraries(evolver ${Boost_LIBRARIES})