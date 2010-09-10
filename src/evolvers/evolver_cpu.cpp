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
#include <fstream>
#include <algorithm>
#include <cmath>
#include "evolvers/evolver_cpu.h"

using namespace std;

void frame_updater::operator()(){
	for(u32 i = start_index; i < end_index; ++i){
		AI::timestep(&aic[i], dt);
		Physics::PhysRunner::timestep(&runners[i], dt);
		TankBullet::update(&bullets[i], dt);
		BasicTank::update(&tanks[i], dt);
		for(u32 j = 0; j < MAX_BULLETS; ++j){
			Collision::bullet_tank_check(&bullets[i],
										 &tanks[i],
										 j);
		}
		for(u32 j = 0; j < MAX_TANKS; ++j){
			Collision::tank_tank_check(&tanks[i], j);
		}
	}
}

void Evolver_cpu::initialize_impl(){
	m_framecount = 0;
	// resize the vectors
	m_runner.resize(NUM_INSTANCES);
	m_bullets.resize(NUM_INSTANCES);
	m_tanks.resize(NUM_INSTANCES);
	m_ai.resize(NUM_INSTANCES);
	
	// setup everything on the CPU
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		Physics::PhysRunner::initialize(&m_runner[i]);
		TankBullet::initialize(&m_bullets[i], &m_runner[i]);
		BasicTank::initialize(&m_tanks[i], &m_runner[i], &m_bullets[i]);
		AI::initialize(&m_ai[i], &m_tanks[i], &m_bullets[i]);
	}
	
	// create a backup copy for quick re-initialization
	m_ai_b = m_ai;
	m_runner_b = m_runner;
	m_tanks_b = m_tanks;
	m_bullets_b = m_bullets;

	for(u32 i = 0; i < MAX_THREADS; ++i){
		m_updaters[i].aic = &m_ai[0];
		m_updaters[i].bullets = &m_bullets[0];
		m_updaters[i].dt = 0;
		m_updaters[i].runners = &m_runner[0];
		m_updaters[i].start_index = i*(NUM_INSTANCES/MAX_THREADS);
		m_updaters[i].end_index = m_updaters[i].start_index +
									(NUM_INSTANCES/MAX_THREADS);
		m_updaters[i].tanks = &m_tanks[0];
	}
}

void Evolver_cpu::cleanup_impl(){
}

void Evolver_cpu::frame_step_impl(float dt){
	for(u32 i = 0; i < MAX_THREADS; ++i){
		// setup the updaters and assign them to threads
		m_updaters[i].dt = dt;
		m_threads[i] = boost::thread(boost::ref(m_updaters[i]));
	}

	for(u32 i = 0; i < MAX_THREADS; ++i){
		// rejoin the threads
		m_threads[i].join();
	}
	++m_framecount;
}

void Evolver_cpu::retrieve_state_impl(){
}

void Evolver_cpu::evolve_ga_impl(){
	// copy over the data to the vector for easy sorting
	vector<pair<u32, u32> > score_data(m_population_score.size());
	for(u32 i = 0; i < score_data.size(); ++i){
		score_data[i].first = i;
		score_data[i].second = m_population_score[i];
	}

	// sort it from highest score to lowest score
	sort(score_data.begin(), score_data.end(), score_sort<u32>);
	if(m_last_score.size() == 0){
		m_last_score = score_data;
	}else{
		bool result = equal(score_data.begin(), score_data.end(),
							m_last_score.begin());
		if(result){
			cout << "GENES DIDN'T EVOLVE" << endl;
		}
		m_last_score = score_data;
	}

	// perform the reproduction process
	// score_data[n].first is the index to the individual
	// second is the score
	for(u32 i = 0; i < score_data.size(); ++i){
		if(i < ELITE_COUNT){
			// copy over the elite genes
			copy_genes(&m_ai_b[i], &m_ai[score_data[i].first]);
		}else{
			// time to reproduce given whatever else there may be
			// we'll force only the 1st half of the set of parents
			u32 p1 = rand() % score_data.size()/2;
			u32 p2 = rand() % score_data.size()/2;
			reproduce(&m_ai_b[i], &m_ai[score_data[p1].first],
					  &m_ai[score_data[p2].first]);

			// random chance to mutate
			u32 m = rand() % 100;
			if(m < MUTATION_RATE){
				mutate(&m_ai_b[i]);
			}
		}
	}
}

u32 Evolver_cpu::retrieve_score_impl(){
	u32 score = 0;
	u32 pos_found = 0;
	for(int i = 0; i < m_population_score.size(); ++i){
		if(m_population_score[i] > score){
			pos_found = i;
			score = m_population_score[i];
		}
	}
	return score;
}

void Evolver_cpu::save_best_gene_impl(const std::string& fname){
	// find the individual with the highest score
	score_map::iterator best_pos;
	u32 score_find = retrieve_score_impl();
	for(best_pos = m_population_score.begin();
	best_pos != m_population_score.end(); ++best_pos){
		if(score_find == best_pos->second){
			break;
		}
	}
	
	ofstream fout(fname.c_str());
	// 	fout.seekp(ios::end);
	
	// assume that we just want AI_CONTROLLER 0
	// write out the accel gene 1st
	if(fout.is_open()){
		u32 index = best_pos->first;
		AI::AI_Core::gene_type tempval;
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			tempval = m_ai[index].gene_accel[i][0];
			fout.write((const char*)&tempval, sizeof(tempval));
		}
		
		// write out the heading gene next
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			tempval = m_ai[index].gene_heading[i][0];
			fout.write((const char*)&tempval, sizeof(tempval));
		}
	}
	
	fout.close();
}

void Evolver_cpu::prepare_game_state_impl(){
	m_population_score.clear();
	m_framecount = 0;

	// restore from the backup buffer
	m_ai = m_ai_b;
	m_runner = m_runner_b;
	m_tanks = m_tanks_b;
	m_bullets = m_bullets_b;

	// setup the stuff on the current buffer
	Physics::vec2 params;
	for(int i = 0; i < NUM_INSTANCES; ++i){
		params.x = -25;
		tank_id evading_tank = BasicTank::spawn_tank(&m_tanks[i],
													 params,
													 90,
													 0);
		AI::add_tank(&m_ai[i], evading_tank, AI_TYPE_EVADER);

		params.x = 3;
		params.y = 12;
		tank_id attacking_tank = BasicTank::spawn_tank(&m_tanks[i],
													   params,
													   180,
													   1);
		AI::add_tank(&m_ai[i], attacking_tank, AI_TYPE_ATTACKER);

		params.x = 20;
		params.y = -12;
		attacking_tank = BasicTank::spawn_tank(&m_tanks[i],
											   params,
											   180,
											   1);
		AI::add_tank(&m_ai[i], attacking_tank, AI_TYPE_ATTACKER);
	}
}

bool Evolver_cpu::is_game_over_impl(){
	//NOTE: we'll also use this function to save their score if needed
	bool all_dead = true;
	for(int i = 0; i < NUM_INSTANCES; ++i){
		// tank 0 is the one dodging, check its status
		all_dead &= (m_tanks[i].state[0] == TANK_STATE_INACTIVE);
		if(m_tanks[i].state[0] == TANK_STATE_INACTIVE
			&& (m_population_score.find(i) == m_population_score.end())){
			// this tank is dead, save the score
			m_population_score[i] = m_framecount;
		}
	}
	return all_dead;
}
