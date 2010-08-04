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
#ifndef UTIL_H
#define UTIL_H
#define PI 3.1415926535897

namespace util{
	
template<typename T>
T degs_to_rads(T degs){
	return (degs * (PI/180));
}

template<typename T>
T rads_to_degs(T rads){
	return (rads * (180/PI));
}
	
};

#endif // UTIL_H