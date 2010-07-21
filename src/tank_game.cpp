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
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <ClanLib/gl.h>
#include <ClanLib/application.h>

class GameDisplay{
public:
	static int main(const std::vector<CL_String>& args);
};

CL_ClanApplication app(&GameDisplay::main);

int GameDisplay::main(const std::vector<CL_String>& args){
	CL_SetupCore setup_core;
	CL_SetupDisplay setup_display;
	CL_SetupGL setup_gl;
	
	try{
		CL_DisplayWindow window("TankWars", 800, 600);
		
		CL_GraphicContext& gc = window.get_gc();
		CL_InputDevice& keyboard = window.get_ic().get_keyboard();
		
		while(!keyboard.get_keycode(CL_KEY_ESCAPE)){
			gc.clear(CL_Colorf::black);
			//TODO: drawing code here
			
			window.flip();
			
			CL_KeepAlive::process();
			//TODO: change this to be based per frame
			CL_System::sleep(10);
		}
	}catch(...){
	}
}