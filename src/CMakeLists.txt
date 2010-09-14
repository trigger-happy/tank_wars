list(APPEND TANKGAME_SRC
			ui/button.cpp
			game_scene/gsmenu.cpp
			game_scene/gslobby.cpp
			game_scene/gsoptions.cpp
			game_scene/gsgame.cu
			tank_game.cpp)
			
list(APPEND EVOLVER_SRC
			data_store/data_store.cpp
			evolvers/evolver_gpu.cu
			evolvers/evolver_cpu.cpp
			evolver.cpp)

if(CUDA_FOUND)
	cuda_add_executable(quick_check qcheck.cu)
	
	cuda_add_executable(tank_game ${TANKGAME_SRC})
	cuda_add_executable(evolver ${EVOLVER_SRC})
	target_link_libraries(evolver ${LINK_LIBS})
	
else(CUDA_FOUND)
	message("CUDA not found, some apps won't be built")
	
	add_executable(tank_game ${TANKGAME_SRC})
	add_executable(evolver ${EVOLVER_SRC})
endif(CUDA_FOUND)

# add_executable(event_replay event_replay.cpp)
# target_link_libraries(event_replay twgame)