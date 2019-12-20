///////////////////////////////////////////////////////////////////////////////
// File:		  GameOfLife.h
// Revision:	  1
// Date Creation: 13.11.2019
// Last Change:	  13.11.2019
// Author:		  Christian Steinbrecher
// Descrition:	  Conway's Game of Life
///////////////////////////////////////////////////////////////////////////////

#include "GameOfLife.h"

#include <math_extended.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void kernel(TElem const dp_fieldOld[], TElem dp_fieldNew[],
size_t const m, size_t const n)
{
	assert(dp_fieldOld != nullptr);
	assert(dp_fieldNew != nullptr);

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= 1 && y < m-1 && x >= 1 && x < n-1)
	{
		dp_fieldNew[y * n + x] = isAlive(dp_fieldOld, m, n, y, x);
	}
}



void gameOfLife_gpu(TElem const dp_fieldOld[], TElem dp_fieldNew[],
	size_t const m, size_t const n)
{
	if (dp_fieldOld == nullptr || dp_fieldNew == nullptr)
	{
		return;
	}


	size_t const block_size = 128;

	unsigned int bigX = static_cast<unsigned int>(ceil_div(n, block_size));
	unsigned int bigY = static_cast<unsigned int>(ceil_div(m, 1));

	unsigned int tibX = static_cast<unsigned int>(block_size);
	unsigned int tibY = static_cast<unsigned int>(1);

	dim3 const big(bigX, bigY);	// blocks in grid
	dim3 const tib(tibX, tibY); // threads in block

	kernel << < big, tib >> > (dp_fieldOld, dp_fieldNew, m, n);
}


__global__ void kernel_drawField(TElem const dp_field[], Vertex dp_vertexBuffer[],
	size_t const height, size_t const width, size_t const strideSize,
	size_t const m, size_t const n)
{
	assert(dp_field != nullptr);
	assert(dp_vertexBuffer != nullptr);

	assert(height * strideSize <= m);
	assert(width * strideSize <= n);

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	size_t const fieldStartX = (n - width * strideSize) / 2;
	size_t const fieldStartY = (m - height * strideSize) / 2;

	if (y < height && x < width)
	{
		if (dp_field[(fieldStartY + y * strideSize) * n + (fieldStartX + x * strideSize)])
		{
			dp_vertexBuffer[y * width + x].color = { 1.0f, 1.0f, 1.0f };
		}
		else
		{
			dp_vertexBuffer[y * width + x].color = { 0.0f, 0.0f, 0.0f };

		}
	}
}


void drawField_gpu(TElem const dp_field[], Vertex dp_vertexBuffer[],
	size_t const height, size_t const width, size_t const strideSize,
	size_t const m, size_t const n)
{
	if (dp_field == nullptr || dp_vertexBuffer == nullptr)
	{
		return;
	}

	if (height * strideSize  > m || width * strideSize > n)
	{
		return;
	}


	size_t const block_size = 128;

	unsigned int bigX = static_cast<unsigned int>(ceil_div(width, block_size));
	unsigned int bigY = static_cast<unsigned int>(ceil_div(height, 1));

	unsigned int tibX = static_cast<unsigned int>(block_size);
	unsigned int tibY = static_cast<unsigned int>(1);

	dim3 const big(bigX, bigY);	// blocks in grid
	dim3 const tib(tibX, tibY); // threads in block

	kernel_drawField << < big, tib >> > (dp_field, dp_vertexBuffer, height, width, strideSize, m, n);
}