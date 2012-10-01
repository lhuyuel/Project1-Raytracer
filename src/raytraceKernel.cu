// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
	ray r;
	r.origin = eye;

	glm::vec3 AVEC,BVEC,MVEC,HVEC,VVEC,Ppoint;//from CIS560 
	float Sx = x / (resolution.x );
	float Sy = y / (resolution.y );
	
	AVEC = glm::cross(view, up);//view is the CVEC, up is UVEC
	BVEC = glm::cross(AVEC, view);
	MVEC = eye + view;//Midpoint of screen
	HVEC =  view.length() * tan(fov.x) * glm::normalize(AVEC); 
	VVEC =  view.length() * tan(fov.y) * glm::normalize(BVEC);
	Ppoint = MVEC + ( 2*Sx - 1 ) * HVEC + ( 2*Sy -1 ) * VVEC; 
	
	r.direction = glm::normalize(Ppoint - eye);
	
	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//generate rays for further ray tracing
__global__ void generateRay(ray *rays, cameraData cam)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	if(x <= cam.resolution.x && y <= cam.resolution.y)
		rays[index] = raycastFromCameraKernel(cam.resolution, 0.0f, x, y, cam.position, cam.view, cam.up, cam.fov);
	__syncthreads();
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, ray* rays)//Added cudaMaterial
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);//pixel
	if(x <= cam.resolution.x && y <= cam.resolution.y)
	{//ray r = raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov);
		ray r = rays[index]; 
		int currentDepth;
		bool continueFlag = true;
		int hitCounter = 0;
		if(currentDepth > 4) return;
		glm::vec3 intersecP, norm, colResult;
		float d = -1;//distance to intersection
		float min_d = 1000000.0;//the distance to closest object
		int closestGeomIndex= -1;
		glm::vec3 lightPos;
		float lightEmi;
		int lightGeoIndex=-1;//store the index of the light source, Only one currently
		float amibient = 0.5, diffuse = 0.5, specular = 0.75;

		for( int i = 0; i < numberOfGeoms; ++i )
		{//this loop find out the closest object and light source
			if(geoms[i].type == SPHERE) d = sphereIntersectionTest(geoms[i],r,intersecP,norm);
			else if(geoms[i].type == CUBE ) d = boxIntersectionTest(geoms[i],r,intersecP,norm);
		
			if( d > 0 && d < min_d )
			{//find out the closest geometry's Index 
				min_d = d;
				closestGeomIndex = i; 
			}
		
			if( (lightEmi = materials[ geoms[i].materialid ].emittance) > 0  )
			{//this object is a light source
				lightPos = geoms[i].translation;
				colResult = materials[ geoms[i].materialid ].color;
				lightGeoIndex = i;
				
				continueFlag = false;
			}
			else lightEmi = 0;
		}
	
		ray light;
		light.origin = intersecP;
		light.direction = glm::normalize(lightPos - intersecP);
		glm::vec3 lightCol = materials[ geoms[lightGeoIndex].materialid ].color;
		light.origin += light.direction*0.1f;//move back 0.1 to change the intersection test
		float dLignt = -1;
		float min_dlight = 10000;
		float fordifuse = glm::dot(norm,light.direction);
		int geoIndex;
		glm::vec3 lightIntersecP, lightIntersecNorm;
		glm::vec3 geoCol;
	
		if(closestGeomIndex >= 0 )
		{//intersection occurred
	
			//if(rayDepth == 0) colResult = glm::vec3(0,0,0);
			if(materials[ geoms[closestGeomIndex].materialid ].emittance > 1) 
			{//the object is light source
				colResult = materials[ geoms[closestGeomIndex].materialid ].color;
				continueFlag = false;//don't need to keep going
			}
			//for(int i = 0; (i < numberOfGeoms) && (i != lightGeoIndex);  ++i)
			else
			{//if the object is not light source object
				geoCol = materials[ geoms[closestGeomIndex].materialid ].color;
				colResult = amibient * materials[ geoms[closestGeomIndex].materialid ].color ;
				for(int i = 0; i < numberOfGeoms;++i )
				{//Shadow light intersection test
					if(geoms[i].type == SPHERE)
						dLignt = sphereIntersectionTest(geoms[i], light, lightIntersecP, lightIntersecNorm);
					else if(geoms[i].type == CUBE)
						dLignt = boxIntersectionTest(geoms[i],light, lightIntersecP, lightIntersecNorm);

					if(dLignt > 0 && dLignt < min_dlight)	
					{
						min_dlight = dLignt;
						geoIndex = i;
					}
				}
				
				if(geoIndex == lightGeoIndex ||geoIndex ==closestGeomIndex )//Only hit the light
				{
					colResult += diffuse * fordifuse ;
					
					if( materials[geoms[closestGeomIndex].materialid].specularExponent != 0 )
					{
						//colResult = glm::vec3(0,0,1);
						glm::vec3 reflectionRay = calculateReflectionDirection(norm,r.direction);
						//specular
						colResult +=  specular * materials[geoms[closestGeomIndex].materialid].specularColor * pow( fabs (glm::dot( r.direction, reflectionRay )), materials[geoms[closestGeomIndex].materialid].specularExponent);
						r.origin = intersecP;
						r.direction = reflectionRay;
					}
				
					if(materials[geoms[closestGeomIndex].materialid].hasReflective != 0)
					{  
						r.direction = calculateReflectionDirection(norm, r.direction);//normalized in function
						r.origin = intersecP + r.direction * 0.01f;
						continueFlag = true;

					}

				}
				//else if( geoIndex ==closestGeomIndex ) 	colResult += diffuse * fordifuse ;
				else colResult=glm::vec3(0,0,0);
				//else colResult=glm::vec3(0,0,0);//test which part goes to "else"
			}
		}
		else 
		{//does not intersect with anything
			colResult= glm::vec3(0.0, 0.0, 1.0);
			continueFlag = false;
		}

	
		colors[index] = colResult;
	}//this is for the if(x <= cam.resolution.x && y <= cam.resolution.y)
}



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 4; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //package materials and sent to GPU
  material* materialsList = new material[numberOfMaterials];
  for(int i=0; i<numberOfMaterials; i++){
	  material newStaticMaterial;
	  newStaticMaterial.color = materials[i].color;
	  newStaticMaterial.specularExponent = materials[i].specularExponent;
	  newStaticMaterial.specularColor = materials[i].specularColor;
	  newStaticMaterial.hasReflective = materials[i].hasReflective;
	  newStaticMaterial.hasRefractive = materials[i].hasRefractive;
	  newStaticMaterial.indexOfRefraction = materials[i].indexOfRefraction;
	  newStaticMaterial.hasScatter = materials[i].hasScatter;
	  newStaticMaterial.absorptionCoefficient = materials[i].absorptionCoefficient;
	  newStaticMaterial.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
	  newStaticMaterial.emittance = materials[i].emittance;
	  materialsList[i] = newStaticMaterial;
  }

  material* cudaMaterials = NULL;
  cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudaMaterials, materialsList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

 //Package rays
 int numOfRays = cam.resolution.x * cam.resolution.y;
 ray *rays = new ray[numOfRays];
 ray *cudarays = NULL;
 cudaMalloc((void**)&cudarays, numOfRays * sizeof(ray));
 cudaMemcpy(cudarays, rays, numOfRays * sizeof(ray), cudaMemcpyHostToDevice);
 
 generateRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudarays,cam);
 //kernel launches
 //traceDepth = 4;
 //for(int i; i < traceDepth; ++i)
 //{
	 raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudaMaterials,cudarays);
 //}
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudaMaterials );
  cudaFree(cudarays);

  delete geomList;
  delete materialsList;
  delete rays;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
