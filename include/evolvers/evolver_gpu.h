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

#ifndef EVOLVER_GPU_H
#define EVOLVER_GPU_H
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <boost/scoped_ptr.hpp>
#include "game_core/basictank.h"
#include "game_core/tankbullet.h"
#include "game_core/tank_ai.h"
#include "evolvers/ievolver.h"

class Evolver_gpu : public iEvolver<Evolver_gpu>{
public:
private:
	friend class iEvolver<Evolver_gpu>;
	void initialize_impl();
	void cleanup_impl();
	void frame_step_impl(f32 dt);
	void retrieve_state_impl();
	void evolve_ga_impl();
	u32 retrieve_score_impl();
	void save_best_gene_impl(const std::string& fname);
	void prepare_game_state_impl();
	void perpare_game_scenario_impl(u32 dist, u32 bullet_loc, u32 bullet_vec);
	void end_game_scenario_impl();
	bool is_game_over_impl();
	void finalize_impl();
	
private:
	// internal functions
	void copy_to_device();
	void copy_from_device();
	
private:
	// for debugging purposes
	std::vector<std::pair<u32, u32> > m_last_score;
	std::vector<u32> m_score;

	// GPU stuff
	Physics::PhysRunner::RunnerCore* m_cuda_runner;
	TankBullet::BulletCollection* m_cuda_bullets;
	BasicTank::TankCollection* m_cuda_tanks;
	AI::AI_Core* m_cuda_ai;
	u32* m_cuda_score;
};

#endif // EVOLVER_GPU_H
