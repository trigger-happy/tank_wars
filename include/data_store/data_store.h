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
#ifndef DATA_STORE_H
#define DATA_STORE_H
#include <string>
#include "data_store/ds_types.h"

namespace kyotocabinet{
	class PolyDB;
}

class DataStore{
public:
	DataStore(const std::string& aidb,
			  const std::string& simdb);
	~DataStore();

	inline bool is_ok() const{
		return m_status;
	}

	bool save_gene_data(const ai_key& key,
						const AI::AI_Core& aic);

	bool get_gene_data(const ai_key& key,
					   AI::AI_Core& aic);

	const char* get_ai_error() const;

	bool save_sim_data(const sim_key& sk,
					   const sim_data& sd);

	bool get_sim_data(const sim_key& sk,
					  sim_data& sd);

	const char* get_sim_error() const;
	
private:
	kyotocabinet::PolyDB* m_aidb;
	kyotocabinet::PolyDB* m_simdb;
	bool m_status;
};

#endif //DATA_STORE_H