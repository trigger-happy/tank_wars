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

#include <vector>
#include <stack>
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <ClanLib/gl.h>
#include <ClanLib/application.h>
#include <boost/lambda/lambda.hpp>
#include <boost/timer.hpp>

#include "game_scene/igamescene.h"

#include "game_scene/gsmenu.h"

// in milliseconds
#define FRAME_TIME 1000.0/60.0

class GameDisplay{
public:
	static int main(const std::vector<CL_String>& args);
	
private:
	static void scene_cleanup();
	
private:
	static boost::timer s_frame_timer;
	static double s_deltatime;
	static std::stack<iGameScene*> s_scene_stack;
};

double					GameDisplay::s_deltatime = 0.0;
boost::timer			GameDisplay::s_frame_timer;
std::stack<iGameScene*>	GameDisplay::s_scene_stack;

// bad design, but no choice for the time being. Keep things simple.
bool					s_running = true;

CL_ClanApplication app(&GameDisplay::main);

void GameDisplay::scene_cleanup(){
	while(!s_scene_stack.empty()){
		delete s_scene_stack.top();
		s_scene_stack.pop();
	}
}

int GameDisplay::main(const std::vector<CL_String>& args){
	CL_SetupCore setup_core;
	CL_SetupDisplay setup_display;
	CL_SetupGL setup_gl;
	
	try{
		CL_DisplayWindow window("TankWars", 800, 600);
		
		CL_GraphicContext& gc = window.get_gc();
		CL_InputDevice& keyboard = window.get_ic().get_keyboard();
		CL_InputDevice& mouse = window.get_ic().get_mouse();
		
		CL_ResourceManager resources("resources/game_resource.xml");
		s_scene_stack.push(new GSMenu(gc, resources));
		
		while(!keyboard.get_keycode(CL_KEY_ESCAPE) && s_running){
			// restart the frame timer
			s_frame_timer.restart();
			
			// clear the screen
			gc.clear(CL_Colorf::black);
			
			if(!s_scene_stack.empty()){
				// use the real delta time if the frame rate is slow
				// otherwise, we hard lock to default frame rate
				double dt = FRAME_TIME;
				if(s_deltatime > FRAME_TIME){
					dt = s_deltatime;
				}
				
				// perform a frame update
				s_scene_stack.top()->onFrameUpdate(dt, &keyboard, &mouse);
				
				// render
				s_scene_stack.top()->onFrameRender(&gc);
			}
			
			// flip screens
			window.flip();
			
			// read windowing messages
			CL_KeepAlive::process();
			
			// get the elapsed frame time
			s_deltatime = s_frame_timer.elapsed()*1000.0;
			
			// sleep until we reach the next frame time iteration
			CL_System::sleep(FRAME_TIME - s_deltatime);
		}
	}catch(CL_Exception& e){
		CL_ConsoleWindow console("error console", 80, 160);
		CL_Console::write_line("Exception: " + e.get_message_and_stack_trace());
		console.display_close_message();
		return -1;
	}
	
	scene_cleanup();
}