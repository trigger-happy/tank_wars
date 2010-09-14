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
#include <kcpolydb.h>
#include "data_store/ds_types.h"

class DataStore{
public:
	DataStore(const std::string& dbname);
	~DataStore();

	inline bool is_open() const{
		return m_status;
	}

	bool save_gene_data(const ai_key& key,
						const AI::AI_Core& aic);

	bool get_gene_data(const ai_key& key,
					   AI::AI_Core& aic);

	inline const char* get_error() const{
		return m_db.error().name();
	}
private:
	kyotocabinet::PolyDB m_db;
	bool m_status;
};

#endif //DATA_STORE_H