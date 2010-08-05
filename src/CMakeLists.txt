list(APPEND TWGAME_SOURCES
			evolvers/evolver_cpu.cpp
			twgame_cpu.cpp)

if(CUDA_FOUND)
	list(APPEND TWGAME_SOURCES
				twgame_gpu.cu
				evolvers/evolver_gpu.cu
				game_core/physics.cu
				game_core/tankbullet.cu
				game_core/basictank.cu)
				
	cuda_add_library(twgame ${TWGAME_SOURCES})
	
	cuda_add_executable(quick_check qcheck.cu)
else(CUDA_FOUND)
	message("CUDA not found, apps that use CUDA will not be built")
	message("This means nearly everything...")
	add_library(twgame ${TWGAME_SOURCES})
endif(CUDA_FOUND)

add_executable(evolver evolver.cpp)
target_link_libraries(evolver twgame)

add_executable(event_replay event_replay.cpp)
target_link_libraries(event_replay twgame)

add_executable(tank_game
				ui/button.cpp
				game_scene/gsmenu.cpp
				game_scene/gslobby.cpp
				game_scene/gsoptions.cpp
				game_scene/gsgame.cpp
				tank_game.cpp)
target_link_libraries(tank_game twgame)