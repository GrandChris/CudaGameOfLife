///////////////////////////////////////////////////////////////////////////////
// File: main.cpp
// Date: 13.11.2019
// Version: 1
// Author: Christian Steinbrecher
// Description: Starts a new game
///////////////////////////////////////////////////////////////////////////////

#include "GameOfLife.h"

#include <ParticleRenderer.h>
#include <CudaPointRenderObject.h>
#include <CudaVertexBuffer.h>

#include <device_info.h>
#include <cuda_check.h>
#include <dp_memory.h>

#include <iostream>
#include <utility>

using namespace std;

int main()
{
	printDeviceInfo();


	// size of the gamegrid
	//size_t const chunkSize = 16;
	size_t const n = 1024 * 32;
	size_t const m = 1024 * 32;
	size_t const elementsCount = m * n;
	size_t strideSize = 16;

	bool const useCPU = false;

	


	auto const hp_field1 = make_unique<TElem[]>(elementsCount);
	auto const hp_field2 = make_unique<TElem[]>(elementsCount);
	auto const dp_field1 = dp_make_unique<TElem[]>(elementsCount);
	auto const dp_field2 = dp_make_unique<TElem[]>(elementsCount);

	TElem * php_field1 = hp_field1.get();
	TElem * php_field2 = hp_field2.get();

	TElem* pdp_field1 = dp_field1.get();
	TElem* pdp_field2 = dp_field2.get();


	size_t const init_m = 16;
	size_t const init_n = 16;
	TElem initField[init_m][init_n] =
	{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};

	size_t const lifeMax_m = 32;
	size_t const lifeMax_n = 32;
	char const lifeMax[lifeMax_m][lifeMax_n+1] = {
		"                                ",
		"                                ",
		"                                ",
		"                     #          ",
		"                    ###         ",
		"               ###    ##        ",
		"              #  ###  # ##      ",
		"             #   # #  # #       ",
		"             #    # # # # ##    ",
		"               #    # #   ##    ",
		"   ####     # #    #   # ###    ",
		"   #   ## # ### ##         ##   ",
		"   #     ##     #               ",
		"    #  ## #  #  # ##            ",
		"          # # # # # #     ####  ",
		"    #  ## #  #  #  ## # ##   #  ",
		"   #     ##   # # #   ##     #  ",
		"   #   ## # ##  #  #  # ##  #   ",
		"   ####     # # # # # #         ",
		"             ## #  #  # ##  #   ",
		"                #     ##     #  ",
		"    ##         ## ### # ##   #  ",
		"     ### #   #    # #     ####  ",
		"     ##   # #    #              ",
		"     ## # # # #    #            ",
		"        # #  # #   #            ",
		"       ## #  ###  #             ",
		"         ##    ###              ",
		"          ###                   ",
		"           #                    ",
		"                                " };
	


	//for (size_t y = 0; y < init_m; ++y)
	//{
	//	for (size_t x = 0; x < init_n; ++x)
	//	{
	//		phpfield1[(y+n/2- init_n/2) * n + (x + n / 2 - init_n / 2)] = initField[y][x];
	//	}
	//}


	for (size_t y = 0; y < lifeMax_m; ++y)
	{
		for (size_t x = 0; x < lifeMax_n; ++x)
		{
			php_field1[(y + m / 2 - lifeMax_m / 2) * n + (x + n / 2 - lifeMax_n / 2)] = lifeMax[y][x] == '#';
		}
	}
	

	// draw
	size_t const width = n / strideSize;
	size_t const height = m / strideSize;
	std::vector<CudaPointRenderObject::Vertex> vertices;
	vertices.resize(width * height);

	for (size_t x = 0; x < width; ++x)
	{
		for (size_t y = 0; y < height; ++y)
		{
			vertices[y * width + x].pos = {
				0.0f,
				static_cast<float>(x) / static_cast<float>(width) - static_cast<float>(0.5f),
				static_cast<float>(y) / static_cast<float>(height)* (static_cast<float>(height) / static_cast<float>(width))
					- static_cast<float>(height) / static_cast<float>(width) / 2.0f
			};

			vertices[y * width + x].color = {
				static_cast<float>(x) / static_cast<float>(width),
				static_cast<float>(y) / static_cast<float>(height),
				1.0f };
		}
	}

	auto app = ParticleRenderer::createVulkan();
	auto obj = CudaPointRenderObject::createVulkan();
	CudaExternalVertexBuffer<CudaPointRenderObject::Vertex> dp_VertexBuffer;


	bool keyZoomInPressed = false;
	bool keyZoomOutPressed = false;
	bool keyHoldPressed = true;
	auto lbd = [&](bool init) {
		// manipulate dp_VertexBuffer
		assert(dp_VertexBuffer.size() == vertices.size());

		static bool firstInit = true;

		if (init == true)
		{
			cudaEvent_t event = { 0 };
			CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));

				CUDA_CHECK(cudaMemcpy(dp_VertexBuffer.get(), vertices.data(),
					dp_VertexBuffer.size() * sizeof(CudaPointRenderObject::Vertex), cudaMemcpyHostToDevice));

				if (firstInit)
				{
					firstInit = false;

					CUDA_CHECK(cudaMemcpy(dp_field1.get(), hp_field1.get(),
						elementsCount * sizeof(TElem), cudaMemcpyHostToDevice));

					CUDA_CHECK(cudaMemcpy(dp_field2.get(), hp_field2.get(),
						elementsCount * sizeof(TElem), cudaMemcpyHostToDevice));
				}

			CUDA_CHECK(cudaEventRecord(event));
			CUDA_CHECK(cudaEventSynchronize(event));
		}


		

		if (useCPU)
		{   // run lokal on the host
			gameOfLife(php_field1, php_field2, m, n);
			std::swap(php_field1, php_field2);


			// display as vertices
			for (size_t x = 0; x < n; ++x)
			{
				for (size_t y = 0; y < m; ++y)
				{
					if (php_field1[y * n + x] > 0)
					{
						vertices[y * n + x].color = { 1.0f, 1.0f, 1.0f };
					}
					else
					{
						vertices[y * n + x].color = { 0.0f, 0.0f, 0.0f };
					}
				}
			}

			cudaEvent_t event = { 0 };
			CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));

			CUDA_CHECK(cudaMemcpy(dp_VertexBuffer.get(), vertices.data(),
				dp_VertexBuffer.size() * sizeof(CudaPointRenderObject::Vertex), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaEventRecord(event));
			CUDA_CHECK(cudaEventSynchronize(event));
		}
		else
		{	// run on the device
			cudaEvent_t event = { 0 };
			CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));

			if (!keyHoldPressed)
			{
				gameOfLife_gpu(pdp_field1, pdp_field2, m, n);
				std::swap(pdp_field1, pdp_field2);
			}

			drawField_gpu(pdp_field1, reinterpret_cast<Vertex*>(dp_VertexBuffer.get()), height, width, strideSize, m, n);

			CUDA_CHECK(cudaEventRecord(event));
			CUDA_CHECK(cudaEventSynchronize(event));
		}
	};

	obj->setVertices(dp_VertexBuffer, vertices.size(), lbd);
	auto pObj = obj.get();
	pObj->setPosition(glm::vec3(4.0f, 0.0f, 0.0f));
	app->add(std::move(obj));

	size_t ultraZoomedIn = 4;
	auto const labda_keyPressed = [&](Key key)
	{
		switch (key)
		{
		case Key::W:
			keyZoomInPressed = true;
			if (strideSize > 1)
			{
				strideSize /= 2;
			}
			else if (ultraZoomedIn > 1)
			{
				ultraZoomedIn /= 2;
				pObj->setPosition(glm::vec3(0.25f * ultraZoomedIn, 0.0f, 0.0f));
			}
			
			break;
		case Key::S:
			keyZoomOutPressed = true;

			if (ultraZoomedIn < 4)
			{
				ultraZoomedIn *= 2;
				pObj->setPosition(glm::vec3(0.25f * ultraZoomedIn, 0.0f, 0.0f));
			}
			else if (strideSize < 16)
			{
				strideSize *= 2;
			}
			break;


		case Key::H:
			keyHoldPressed = true;
			break;

		default:
			break;
		}
	};
	auto const labda_keyReleased = [&](Key key)
	{
		switch (key)
		{
		case Key::W:
			keyZoomInPressed = false;
			break;
		case Key::S:
			keyZoomOutPressed = false;
			break;


		case Key::H:
			keyHoldPressed = false;
			break;

		default:
			break;
		}
	};
	app->keyPressed.add(labda_keyPressed);
	app->keyReleased.add(labda_keyReleased);

	//app->setVSync(true);
	app->run();
	

	return 0;
}