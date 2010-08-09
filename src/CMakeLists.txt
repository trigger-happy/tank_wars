list(APPEND TWGAME_SOURCES
			game_core/twgame.cu)

if(CUDA_FOUND)
	cuda_add_library(twgame ${TWGAME_SOURCES})
	
	cuda_add_executable(quick_check qcheck.cu)
	
else(CUDA_FOUND)
	message("CUDA not found, some apps won't be built")
	
	add_library(twgame ${TWGAME_SOURCES})
endif(CUDA_FOUND)

# add_executable(event_replay event_replay.cpp)
# target_link_libraries(event_replay twgame)

add_executable(tank_game
				ui/button.cpp
				game_scene/gsmenu.cpp
				game_scene/gslobby.cpp
				game_scene/gsoptions.cpp
				game_scene/gsgame.cu
				tank_game.cpp)
target_link_libraries(tank_game twgame)