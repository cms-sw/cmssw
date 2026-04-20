#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"

#include "RecoVertex/PrimaryVertexProducer/plugins/alpaka/DAInBlocksClusterizerAlgo.h"

#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR 1

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  //////////////////////
  // Device functions //
  //////////////////////

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC static void resortVerticesAndAssign(const TAcc& acc,
                                                    TrackDeviceCollection::View tracks,
                                                    VertexDeviceCollection::View vertices,
                                                    ClusterParameters const& cParams) {
    // Multiblock vertex arbitration
    double beta = 1. / cParams.Tstop;
    const unsigned int maxVerticesInSoA = 1024;
    int blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  
    auto& z = alpaka::declareSharedVar<float[maxVerticesInSoA], __COUNTER__>(acc);
    auto& rho = alpaka::declareSharedVar<float[maxVerticesInSoA], __COUNTER__>(acc);
    alpaka::syncBlockThreads(acc);
    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::resortVerticesAndAssign()] Start reassignment\n");
        }
    #endif

    for (unsigned int iv = 0; iv < maxVerticesInSoA; iv += blockSize) {
        if (vertices[iv].isGood()){
	    if ((vertices[iv].rho() > 10000) || (abs(vertices[iv].z()) > 30)){
	        vertices[iv].isGood() = false;    
	    }
	}
    }

    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::resortVerticesAndAssign()] Reorder vertices\n");
        }
    #endif

    alpaka::syncBlockThreads(acc);
    if (once_per_block(acc)){
	// We only keep the ones that are good out of all of the ones in the collection
	unsigned int nTrueVertex = 0;
        for (unsigned int ivO = 0; ivO < maxVerticesInSoA; ivO += 1) {
	    int iv = vertices[ivO].order();
	    if ((iv < 0) || (iv >= (int) maxVerticesInSoA)) continue; 
            if (vertices[iv].isGood()){
                z[nTrueVertex] = vertices[iv].z();
                rho[nTrueVertex] = vertices[iv].rho();
                nTrueVertex++;
          }
        }
        vertices[0].nV() = nTrueVertex;
    }
    alpaka::syncBlockThreads(acc);
    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::resortVerticesAndAssign()] Before vertices are reordered, nV %i left\n", vertices[0].nV());
        }
    #endif

    auto& orderedIndices = alpaka::declareSharedVar<uint16_t[maxVerticesInSoA], __COUNTER__>(acc);
    auto& sws = alpaka::declareSharedVar<uint16_t[maxVerticesInSoA], __COUNTER__>(acc);

    cms::alpakatools::radixSort<Acc1D, float, 2>(acc, z, orderedIndices, sws, vertices[0].nV());
    alpaka::syncBlockThreads(acc);
    // copy sorted vertices back to the SoA. We restrict our usage to the first vertices[0].nV() entries of the SoA
    for (int ivtx = threadIdx; ivtx < vertices[0].nV(); ivtx += blockSize) {
        vertices[ivtx].z() = z[ivtx];
        vertices[ivtx].rho() = rho[ivtx];
        vertices[ivtx].order() = orderedIndices[ivtx];
	vertices[ivtx].isGood() = true;
    }
    // And invalidate the remaining part we won't use anymore
    for (unsigned int ivtx = vertices[0].nV()+threadIdx; ivtx < maxVerticesInSoA; ivtx += blockSize) {
        vertices[ivtx].isGood() = false;
    }
    alpaka::syncBlockThreads(acc);
    double zrange_min_ = 0.1;

    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::resortVerticesAndAssign()] Vertices are reordered, nV %i left\n", vertices[0].nV());
        }
    #endif


    for (int itrack = threadIdx; itrack < tracks.nT(); itrack += blockSize) {
      if (not(tracks[itrack].isGood()))
        continue;
      double zrange = std::max(cParams.zrange / sqrt((beta)*tracks[itrack].oneoverdz2()), zrange_min_);
      double zmin = tracks[itrack].z() - zrange;
      int kmin = vertices[0].nV() - 1;
      if (kmin < 0)
        continue;
      if (vertices[vertices[kmin].order()].z() > zmin) {  
        while ((kmin > 0) &&
               (vertices[vertices[kmin - 1].order()].z() >
                zmin)) {  
          // i.e., while we find another vertex within range that is before the previous initial step we select it as minimum
          kmin--;
        }
      } else {
        while ((kmin < vertices[0].nV()) &&
               (vertices[vertices[kmin].order()].z() <
                zmin)) {  
          kmin++;
        }
      }
      // Now the same for the upper bound
      double zmax = tracks[itrack].z() + zrange;
      int kmax = 0;
      if (vertices[vertices[kmax].order()].z() < zmax) {
        while (
            (kmax < vertices[0].nV() - 1) &&
            (vertices[vertices[kmax + 1].order()].z() <
             zmax)) {  
	   // As long as we have more vertex above kmax but within z range, we can add them to the collection, keep going
          kmax++;
        }
      } else {  
        while ((kmax > 0) && (vertices[vertices[kmax].order()].z() > zmax)) {
          kmax--;
        }
      }
      if (kmin <= kmax) {
        tracks[itrack].kmin() = kmin;
        tracks[itrack].kmax() = kmax + 1;  
      } else {
        tracks[itrack].kmin() = std::max(0, std::min(kmin, kmax));
        tracks[itrack].kmax() = std::min(vertices[0].nV(), std::max(kmin, kmax) + 1);
      }
    }
    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::resortVerticesAndAssign()] Track-vertex range assignment finished \n");
        }
    #endif

    alpaka::syncBlockThreads(acc);
    double mintrkweight_ = 0.5;
    double rho0 = vertices[0].nV() > 1 ? 1. / vertices[0].nV() : 1.;
    double z_sum_init = rho0 * exp(-(beta)*cParams.dzCutOff * cParams.dzCutOff);
    for (int itrack = threadIdx; itrack < tracks.nT(); itrack += blockSize) {
      int kmin = tracks[itrack].kmin();
      int kmax = tracks[itrack].kmax();
      double p_max = -1;
      int iMax = 10000;
      double sum_Z = z_sum_init;
      for (auto k = kmin; k < kmax; k++) {
        double v_exp = exp(-(beta)*std::pow(tracks[itrack].z() - vertices[vertices[k].order()].z(), 2) *
                           tracks[itrack].oneoverdz2());
        sum_Z += vertices[vertices[k].order()].rho() * v_exp;
      }
      double invZ = sum_Z > 1e-40 ? 1. / sum_Z : 0.0;
      // Univocally assign a track to the highest probability vertex
      for (auto k = kmin; k < kmax; k++) {
        float v_exp = exp(-(beta)*std::pow(tracks[itrack].z() - vertices[vertices[k].order()].z(), 2) *
                          tracks[itrack].oneoverdz2());
        float p = vertices[vertices[k].order()].rho() * v_exp * invZ;
        if (p > p_max && p > mintrkweight_) {
          // assign  track i -> vertex k (hard assignment unles we configure a non standard mintrweight_
          p_max = p;
          iMax = k;
        }
      }
      // Finally register that itrack is assigned to vertex iMax
      tracks[itrack].kmin() = iMax;
      tracks[itrack].kmax() = iMax + 1;
    }
    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::resortVerticesAndAssign()] Updated vertex masses\n");
        }
    #endif

    alpaka::syncBlockThreads(acc);
  }  //resortVerticesAndAssign

  template <bool debug = false, typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC static void finalizeVertices(const TAcc& acc,
                                             TrackDeviceCollection::View tracks,
                                             VertexDeviceCollection::View vertices,
                                             ClusterParameters const& cParams) {
    int blockSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    int threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; 
    // First put the tracks in vertex SoA
    for (int k = threadIdx; k < vertices[0].nV(); k += blockSize) { 
        int ivertex = vertices[k].order();
        vertices[ivertex].ntracks() = 0;
        for (int itrack = 0; itrack < tracks.nT(); itrack += 1) {
          if (not(tracks[itrack].isGood())){
            continue;
	  }
          int ivtxFromTk = tracks[itrack].kmin();
          if (ivtxFromTk == k) {
            bool isNew = true;
            for (int ivtrack = 0; ivtrack < vertices[ivertex].ntracks(); ivtrack++) {
              if (tracks[itrack].tt_index() == tracks[vertices[ivertex].track_id()[ivtrack]].tt_index())
                isNew = false;
            }
            if (!isNew){
              continue;
	    }
	    // Here we are saving the itrack (i.e. index in the device collection) instead tracks[itrack].tt_index(); (index of the input reco track collection) so we can use the same device collection in the fitter
            vertices[ivertex].track_id()[vertices[ivertex].ntracks()] = itrack;  
            vertices[ivertex].track_weight()[vertices[ivertex].ntracks()] = 1.;
            vertices[ivertex].ntracks()++;
          }
        }
        if (vertices[ivertex].ntracks() < 2) {
          vertices[ivertex].isGood() = false;  
          continue;                            
        }
        vertices[ivertex].x() = 0;
        vertices[ivertex].y() = 0;
    }
    alpaka::syncBlockThreads(acc);
    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::finalizeVertices()] Before cleanup %i vertices\n", vertices[0].nV());
        }
    #endif
    if (once_per_block(acc)) {
      // So we now check whether each vertex is further enough from the previous one
      for (int k = 0; k < vertices[0].nV(); k++) {
        int prevVertex = ((int)k) - 1;
        int thisVertex = (int)vertices[k].order();
        if (not(vertices[thisVertex].isGood())) {
          continue;
        }
        while (prevVertex >= 0) {
          // Find the previous vertex that was good
          if (vertices[vertices[prevVertex].order()].isGood())
            break;  //Can't be part of the while condition as otherwise it could try to look up with index -1 in the vervex view
          prevVertex--;
        }
        if ((prevVertex < 0)) {  
          // If it is first, always good
          vertices[thisVertex].isGood() = true;
	  #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
              if (once_per_block(acc)) {
                  printf("[DAInBlocksClusterizerAlgoArbitrator::finalizeVertices()] Vertex %i (%i) is good (first)\n", thisVertex, k);
              }
          #endif
        } else if (abs(vertices[thisVertex].z() - vertices[vertices[prevVertex].order()].z()) >
                   (2 * cParams.vertexSize)) {  
	  // If it is further away enough, it is also good
          vertices[thisVertex].isGood() = true;
          #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
              if (once_per_block(acc)) {
                  printf("[DAInBlocksClusterizerAlgoArbitrator::finalizeVertices()] Vertex %i (%i) is good (far away from previous), %1.5f, %1.5f, %1.5f\n", thisVertex, k, vertices[thisVertex].z(), vertices[vertices[prevVertex].order()].z(), 2 * cParams.vertexSize);
              }
          #endif
        } else {
          #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
              if (once_per_block(acc)) {
                  printf("[DAInBlocksClusterizerAlgoArbitrator::finalizeVertices()] Vertex %i (%i) is bad (close to previous), %1.5f, %1.5f\n", thisVertex, k, vertices[thisVertex].z(), vertices[vertices[prevVertex].order()].z());
              }
          #endif
          vertices[thisVertex].isGood() = false;
	  int prevVertexO = vertices[prevVertex].order();
          for (int itrack = 0; itrack < vertices[thisVertex].ntracks(); itrack += 1) {
              bool isNew = true;
              for (int ivtrack = 0; ivtrack < vertices[prevVertexO].ntracks(); ivtrack++) {
                if ( vertices[prevVertexO].track_id()[ivtrack] == vertices[thisVertex].track_id()[itrack] )
                  isNew = false;
              }
              if (!isNew) continue; 
	      vertices[prevVertexO].track_id()[vertices[prevVertexO].ntracks()] = vertices[thisVertex].track_id()[itrack];
	      vertices[prevVertexO].track_weight()[vertices[prevVertexO].ntracks()] = 1.;
	      vertices[prevVertexO].ntracks()++;
          }
        }
      }
      // We have to deal with the order being broken by the invalidation of vertexes and set back again the vertex multiplicity, unfortunately can't be parallelized without threads competing
      int k = 0;
      while (k != vertices[0].nV()) {
        int thisVertex = vertices[k].order();
	printf("%i, %i, %i\n", k,thisVertex, vertices[0].nV());
	if (thisVertex == 9999) { 
	  // i.e. if it was purged it was bad
          for (int l = k; l < vertices[0].nV(); l++) {  
            vertices[l].order() = vertices[l + 1].order();
          }
          vertices[0].nV()--;  
	}
	else if (vertices[thisVertex].isGood()) {  // If is good just continue
          k++;
        } else {
          for (int l = k; l < vertices[0].nV(); l++) {  //If it is bad, move one position all indexes
            vertices[l].order() = vertices[l + 1].order();
          }
          vertices[0].nV()--;  // And reduce vertex number by 1
        }
      }
    }
    #ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
        if (once_per_block(acc)) {
            printf("[DAInBlocksClusterizerAlgoArbitrator::finalizeVertices()] Vertices are reordered, nV %i left\n", vertices[0].nV());
        }
    #endif
    alpaka::syncBlockThreads(acc);
  }  //finalizeVertices

  class ArbitrateKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TrackDeviceCollection::View tracks,
                                  VertexDeviceCollection::View vertices,
                                  ClusterParameters const& cParams,
                                  int32_t nBlocks) const {
      // This has the core of the clusterization algorithm
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoArbitrator::operator()] Start arbitration\n");
      }
#endif
      resortVerticesAndAssign(acc, tracks, vertices, cParams);
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoArbitrator::operator()] Vertex reassignment finished\n");
      }
#endif
      alpaka::syncBlockThreads(acc);
      finalizeVertices(acc, tracks, vertices, cParams);  // In CUDA it used to be verticesAndClusterize
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ARBITRATOR
      if (once_per_block(acc)) {
        printf("[DAInBlocksClusterizerAlgoArbitrator::operator()] Vertices finalized\n");
      }
#endif
      alpaka::syncBlockThreads(acc);
    }
  };  // class kernel

  void DAInBlocksClusterizerAlgo::arbitrate(Queue& queue,
                                  TrackDeviceCollection& deviceTrack,
                                  VertexDeviceCollection& deviceVertex,
				  ClusterParameters const& cParams,
                                  int32_t nBlocks,
                                  int32_t blockSize) {
    const int blocks = divide_up_by(blockSize, blockSize);  //Single block, as it has to converge to a single collection
    alpaka::exec<Acc1D>(
        queue,
        make_workdiv<Acc1D>(blocks, blockSize),
        ArbitrateKernel{},
        deviceTrack
            .view(),
        deviceVertex.view(),
        cParams,
        nBlocks);
  }  // arbitraterAlgo::arbitrate

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
