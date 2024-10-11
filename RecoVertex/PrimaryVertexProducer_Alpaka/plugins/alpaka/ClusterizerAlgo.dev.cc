#include "RecoVertex/PrimaryVertexProducer_Alpaka/plugins/alpaka/ClusterizerAlgo.dev.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  ////////////////////// 
  // Device functions //
  //////////////////////

  class clusterizeKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,  portablevertex::TrackDeviceCollection::View tracks, portablevertex::VertexDeviceCollection::View vertices, const portablevertex::ClusterParamsHostCollection::ConstView cParams, double *beta_, double *osumtkwt_) const{ 
      // This has the core of the clusterization algorithm
      // First, declare beta=1/T
      printf("initialize start\n");
      initialize(acc, tracks, vertices, cParams);
      printf("initialize end\n");
      int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread number inside block
      int blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; // Block number inside grid

      double& _beta = alpaka::declareSharedVar<double, __COUNTER__>(acc);
      double& osumtkwt = alpaka::declareSharedVar<double, __COUNTER__>(acc);
      for (int itrack = threadIdx+blockIdx*blockSize; itrack < threadIdx+(blockIdx+1)*blockSize ; itrack += blockSize){ // TODO:Saving and reading in the tracks dataformat might be a bit too much?
	double temp_weight = static_cast<double>(tracks[itrack].weight());      
        //alpaka::atomicAdd(acc, &osumtkwt, static_cast<double&>(tracks[itrack].weight()), alpaka::hierarchy::Threads{});
	alpaka::atomicAdd(acc, &osumtkwt, temp_weight, alpaka::hierarchy::Threads{});
      }
      alpaka::syncBlockThreads(acc);     
      // In each block, initialize to a single vertex with all tracks
      initialize(acc, tracks, vertices, cParams);
      alpaka::syncBlockThreads(acc);
      printf("Reinitialize\n");
      // First estimation of critical temperature
      getBeta0(acc, tracks, vertices, cParams, _beta);
      alpaka::syncBlockThreads(acc);
      printf("Beta0\n");
      // Cool down to beta0 with rho = 0.0 (no regularization term)
      thermalize(acc, tracks, vertices, cParams, osumtkwt, _beta, cParams.delta_highT(), 0.0);
      alpaka::syncBlockThreads(acc);
      printf("Thermalize\n");
      // Now the cooling loop
      coolingWhileSplitting(acc, tracks, vertices, cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
      printf("Cooling\n");
      // After cooling, merge closeby vertices
      reMergeTracks(acc,tracks, vertices,cParams, osumtkwt, _beta);
      alpaka::syncBlockThreads(acc);
      printf("Merge\n");
      if (once_per_block(acc)){
        beta_[blockIdx]     = _beta;
        osumtkwt_[blockIdx] = osumtkwt;
      }
      alpaka::syncBlockThreads(acc); 
      printf("End\n");
    }
  }; // class kernel
 
  ClusterizerAlgo::ClusterizerAlgo(Queue& queue, int32_t bSize) : 
	  beta_(cms::alpakatools::make_device_buffer<double[]>(queue, bSize)),
	  osumtkwt_(cms::alpakatools::make_device_buffer<double[]>(queue, bSize))  {
    alpaka::memset(queue,  beta_, bSize);
    alpaka::memset(queue,  osumtkwt_, bSize);
  }  

  void ClusterizerAlgo::clusterize(Queue& queue, portablevertex::TrackDeviceCollection& deviceTrack, portablevertex::VertexDeviceCollection& deviceVertex, const std::shared_ptr<portablevertex::ClusterParamsHostCollection> cParams, int32_t nBlocks, int32_t blockSize){
    const int blocks = divide_up_by(nBlocks*blockSize, blockSize); //nBlocks of size blockSize
    alpaka::exec<Acc1D>(queue,
		        make_workdiv<Acc1D>(blocks, blockSize),
			clusterizeKernel{},
			deviceTrack.view(), // TODO:: Maybe we can optimize the compiler by not making this const? Tracks would not be modified
			deviceVertex.view(),
			cParams->view(),
                        beta_.data(),
                        osumtkwt_.data()
                        );			
  } // ClusterizerAlgo::clusterize
} // namespace ALPAKA_ACCELERATOR_NAMESPACE
