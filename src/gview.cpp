/*
	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Library General Public
	License version 2 as published by the Free Software Foundation.

	This library is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
	Library General Public License for more details.

	You should have received a copy of the GNU Library General Public License
	along with this library; see the file COPYING.LIB.  If not, write to
	the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
	Boston, MA 02110-1301, USA.
*/

#include <iostream>
#include <vector>
#include <stack>
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <ClanLib/gl.h>
#include <ClanLib/application.h>
#include <boost/lambda/lambda.hpp>
#include <boost/timer.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/program_options.hpp>
#include "game_display.h"
#include "types.h"

#include "game_scene/igamescene.h"

#include "game_scene/gsgene_view.h"

#include "data_store/data_store.h"

// in milliseconds
#define FRAME_TIME 1000.0f/60.0f

double					GameDisplay::s_deltatime = 0.0;
boost::timer			GameDisplay::s_frame_timer;
std::stack<iGameScene*>	GameDisplay::s_scene_stack;
bool					GameDisplay::s_running = true;
bool					GameDisplay::s_usecuda = false;
bool					GameDisplay::s_view_gene = false;
u32						g_dist = 1;
u32						g_vect = 0;
u32						g_sect = 0;

// This is a prototype, deal with the fact we're using global vars
DataStore* g_db = NULL;
ai_key g_aik;

namespace po = boost::program_options;
using namespace std;

void GameDisplay::push_scene(iGameScene* scene){
	if(!s_scene_stack.empty()){
		s_scene_stack.top()->onSceneDeactivate();
	}
	s_scene_stack.push(scene);
	scene->onSceneActivate();
}

void GameDisplay::pop_scene(){
	if(!s_scene_stack.empty()){
		s_scene_stack.top()->onSceneDeactivate();
		s_scene_stack.pop();
	}
	if(!s_scene_stack.empty()){
		s_scene_stack.top()->onSceneActivate();
	}
}

int GameDisplay::main(){
	CL_SetupCore setup_core;
	CL_SetupDisplay setup_display;
	CL_SetupGL setup_gl;
	
	try{
		u32 score;
		CL_StringFormat fmt("GV ID: %1 GEN: %2 SCORE: %3");
		CL_DisplayWindow window("GENE VIEWER", 800, 600);
		
		CL_GraphicContext& gc = window.get_gc();
		CL_InputDevice& keyboard = window.get_ic().get_keyboard();
		CL_InputDevice& mouse = window.get_ic().get_mouse();
		
		CL_ResourceManager resources("resources/game_resource.xml");
		GSGeneView* gv = new GSGeneView(gc, resources);
		s_scene_stack.push(gv);

		std::vector<u32> scenario_results;

		g_db->get_gene_data(g_aik, score, gv->get_ai(), scenario_results);
		fmt.set_arg(1, g_aik.id);
		fmt.set_arg(2, g_aik.generation);
		fmt.set_arg(3, score);
		window.set_title(fmt.get_result());
		
		while(!keyboard.get_keycode(CL_KEY_ESCAPE) && s_running){
			// restart the frame timer
			s_frame_timer.restart();
			
			// clear the screen
			gc.clear(CL_Colorf::black);
			
			if(!s_scene_stack.empty()){
				// use the real delta time if the frame rate is slow
				// otherwise, we hard lock to default frame rate
				f32 dt = FRAME_TIME;
				/*if(s_deltatime > FRAME_TIME){
					dt = s_deltatime;
				}*/
				
				// perform a frame update
				s_scene_stack.top()->onFrameUpdate(dt, &keyboard, &mouse);
				
				// render
				s_scene_stack.top()->onFrameRender(&gc);
			}
			
			// flip screens
			window.flip();
			
			// read windowing messages
			CL_KeepAlive::process();
			
			// get the elapsed frame time, it's now in milliseconds
			s_deltatime = s_frame_timer.elapsed()*1000;
			
			// sleep until we reach the next frame time iteration
			//NOTE: this seems to be buggy in windows
#if !defined(WIN32)
			CL_System::sleep(FRAME_TIME - s_deltatime);
#else
			//Sleep(FRAME_TIME - s_deltatime);
#endif
		}
		delete s_scene_stack.top();
		s_scene_stack.pop();
	}catch(CL_Exception& e){
		CL_ConsoleWindow console("error console", 80, 160);
		CL_Console::write_line("Exception: " + e.get_message_and_stack_trace());
		console.display_close_message();
		return -1;
	}
}

int main(int argc, char* argv[]){
	srand(std::time(NULL));
	po::options_description desc("Gene viewer options");
	desc.add_options()
		("help", "Show this help message")
		("use-cuda", "Use CUDA")
		("generation", po::value<u32>(), "The generation number")
		("id", po::value<u32>(), "The id of the gene to view")
		("db", po::value<std::string>(), "The gene database, default is genes.kch")
		("dist", po::value<u32>(), "The distance state to check")
		("sect", po::value<u32>(), "The sector state to check")
		("vect", po::value<u32>(), "The vector state to check");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << desc << std::endl;
		return 0;
	}

	if(vm.count("dist")){
		g_dist = vm["dist"].as<u32>();
	}

	if(vm.count("sect")){
		g_sect = vm["sect"].as<u32>();
	}

	if(vm.count("vect")){
		g_vect = vm["vect"].as<u32>();
	}

	if(vm.count("use-cuda")){
		GameDisplay::s_usecuda = true;
	}

	// set the default values
	g_aik.id = 0;
	g_aik.generation = 1;
	std::string aidb = "genes.kch";
	std::string simdb = "simulation.kch";

	if(vm.count("generation")){
		g_aik.generation = vm["generation"].as<u32>();
	}

	if(vm.count("id")){
		g_aik.id = vm["id"].as<u32>();
	}

	if(vm.count("db")){
		aidb = vm["db"].as<std::string>();
	}

	g_db = new DataStore(aidb, simdb);
	GameDisplay::main();
	delete g_db;
}