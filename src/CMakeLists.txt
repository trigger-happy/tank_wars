list(APPEND TWGAME_SOURCES
			evolvers/evolver_cpu.cpp
			twgame_cpu.cpp)

if(CUDA_FOUND)
	list(APPEND TWGAME_SOURCES
				twgame_gpu.cu
				evolvers/evolver_gpu.cu)
				
	cuda_add_library(twgame ${TWGAME_SOURCES})
	
	cuda_add_executable(quick_check qcheck.cu)
else(CUDA_FOUND)
	message("CUDA not found, apps that use CUDA will not be built")
	add_library(twgame ${TWGAME_SOURCES})
endif(CUDA_FOUND)

add_executable(evolver evolver.cpp)
target_link_libraries(evolver twgame)

add_executable(event_replay event_replay.cpp)
target_link_libraries(event_replay twgame)

add_executable(tank_game
				game_scene/gsmenu.cpp
				tank_game.cpp)