list(APPEND TANKGAME_SRC
			ui/button.cpp
			game_scene/gsmenu.cpp
			game_scene/gslobby.cpp
			game_scene/gsoptions.cpp
			game_scene/gsgame.cu
			tank_game.cpp)

list(APPEND GVIEW_SRC
			gview.cpp
			data_store/data_store.cpp
			game_scene/gsgene_view.cu)

list(APPEND SIMVIEW_SRC
			sview.cpp
			data_store/data_store.cpp
			game_scene/gssim_view.cu)
			
list(APPEND EVOLVER_SRC
			data_store/data_store.cpp
			evolvers/evolver_gpu.cu
			evolvers/evolver_cpu.cpp
			evolver.cpp)

if(CUDA_FOUND)
	cuda_add_executable(quick_check qcheck.cu)
	
	cuda_add_executable(tank_game ${TANKGAME_SRC})
	cuda_add_executable(evolver ${EVOLVER_SRC})
	cuda_add_executable(simview ${SIMVIEW_SRC})
	cuda_add_executable(geneview ${GVIEW_SRC})
	target_link_libraries(evolver ${LINK_LIBS})
	target_link_libraries(geneview ${LINK_LIBS})
	
else(CUDA_FOUND)
	message("CUDA not found, some apps won't be built")
	
	add_executable(tank_game ${TANKGAME_SRC})
	add_executable(evolver ${EVOLVER_SRC})
	add_executable(simview ${SIMVIEW_SRC})
	add_executable(geneview ${GVIEW_SRC})
endif(CUDA_FOUND)

# add_executable(event_replay event_replay.cpp)
# target_link_libraries(event_replay twgame)