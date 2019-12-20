///////////////////////////////////////////////////////////////////////////////
// File:		  GameOfLife.h
// Revision:	  1
// Date Creation: 13.11.2019
// Last Change:	  13.11.2019
// Author:		  Christian Steinbrecher
// Descrition:	  Conway's Game of Life
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined __CUDACC__
#define GPU_ENABLED __device__ __forceinline__
#else
#define GPU_ENABLED inline
#endif

#include <glm/glm.hpp>

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;
};

using TElem = unsigned char;

// Calculates an iteration of the game
GPU_ENABLED void gameOfLife(TElem const hp_fieldOld[], TElem hp_fieldNew[], 
	size_t const m, size_t const n);

// Calculates if the given cell is in the next iteration alive
GPU_ENABLED bool isAlive(TElem const field[], size_t const m, 
	size_t const n, size_t const y, size_t const x);

// Counts the alive neighbours of the element
GPU_ENABLED TElem countNeighbours(TElem const field[], 
	size_t const m, size_t const n, size_t const y, size_t const x);


// Functions on the gpu
void gameOfLife_gpu(TElem const dp_fieldOld[], TElem dp_fieldNew[], 
	size_t const m, size_t const n);


void drawField_gpu(TElem const dp_field[], Vertex dp_vertexBuffer[], 
	size_t const height, size_t const width, size_t const strideSize, 
	size_t const m, size_t const n);





// #######+++++++ Implementation +++++++#######

#include <cassert>

GPU_ENABLED void gameOfLife(TElem const fieldOld[], TElem fieldNew[], size_t const m, size_t const n)
{
	for (size_t x = 1; x < n - 1; ++x)
	{
		for (size_t y = 1; y < m - 1; ++y)
		{
			fieldNew[y * n + x] = isAlive(fieldOld, m, n, y, x);
		}
	}
}


GPU_ENABLED bool isAlive(TElem const field[], size_t const m, size_t const n, size_t const y, size_t const x)
{
	assert(x >= 1 && x < n - 1);
	assert(y >= 1 && y < m - 1);

	auto const neighbours = countNeighbours(field, m, n, y, x);

	if (neighbours == 3)
	{
		return true;
	}
	else if (neighbours > 3 || neighbours < 2)
	{
		return false;
	}
	else
	{
		return field[y * n + x];
	}
}


GPU_ENABLED TElem countNeighbours(TElem const field[], size_t const m, size_t const n, size_t const y, size_t const x)
{
	assert(x >= 1 && x < n - 1);
	assert(y >= 1 && y < m - 1);

	return field[(y - 1) * n + (x - 1)] + field[(y - 1) * n + (x + 0)] + field[(y - 1) * n + (x + 1)] +
		   field[(y + 0) * n + (x - 1)] +                                field[(y + 0) * n + (x + 1)] +
		   field[(y + 1) * n + (x - 1)] + field[(y + 1) * n + (x + 0)] + field[(y + 1) * n + (x + 1)];
}







