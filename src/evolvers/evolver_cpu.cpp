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
#include <cmath>
#include "evolvers/evolver_cpu.h"


void Evolver_cpu::initialize_impl(){
}

void Evolver_cpu::cleanup_impl(){
}

void Evolver_cpu::frame_step_impl(float dt){
}

void Evolver_cpu::retrieve_state_impl(){
}

void Evolver_cpu::evolve_ga_impl(){
}

u32 Evolver_cpu::retrieve_score_impl(){
	return 0;
}

void Evolver_cpu::save_best_gene_impl(const std::string& fname){
}

void Evolver_cpu::prepare_game_state_impl(){
}

bool Evolver_cpu::is_game_over_impl(){
	return false;
}
