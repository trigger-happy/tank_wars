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
#ifndef DS_TYPES_H
#define DS_TYPES_H
#include "game_core/tank_ai.h"
#include "game_core/physics.h"
#include "game_core/basictank.h"
#include "game_core/tankbullet.h"
#include "types.h"

#define MAX_BODY_RECORD 18000

// for the AI records
struct ai_key{
	u32 id;
	u32 generation;
};

struct ai_data{
	u32 score;
	AI::AI_Core::gene_type gene_accel[MAX_GENE_DATA][MAX_AI_EVADERS];
	AI::AI_Core::gene_type gene_heading[MAX_GENE_DATA][MAX_AI_EVADERS];
};

// for the simulation data
struct sim_key{
	u32 id;
	u32 generation;
};

struct sim_data{
	BasicTank::TankCollection tc;
	TankBullet::BulletCollection bc;
	Physics::physBody bodies[MAX_BODY_RECORD];
};

#endif //DS_TYPES_H