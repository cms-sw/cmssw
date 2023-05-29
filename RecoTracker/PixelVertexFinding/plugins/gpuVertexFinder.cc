#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"

#include "PixelVertexWorkSpaceUtilities.h"
#include "PixelVertexWorkSpaceSoAHost.h"
#include "PixelVertexWorkSpaceSoADevice.h"

#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

#undef PIXVERTEX_DEBUG_PRODUCE

namespace gpuVertexFinder {

  // reject outlier tracks that contribute more than this to the chi2 of the vertex fit
  constexpr float maxChi2ForFirstFit = 50.f;
  constexpr float maxChi2ForFinalFit = 5000.f;

  // split vertices with a chi2/NDoF greater than this
  constexpr float maxChi2ForSplit = 9.f;

  template <typename TrackerTraits>
  __global__ void loadTracks(
      TrackSoAConstView<TrackerTraits> tracks_view, VtxSoAView soa, WsSoAView pws, float ptMin, float ptMax) {
    auto const* quality = tracks_view.quality();
    using helper = TracksUtilities<TrackerTraits>;
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = first, nt = tracks_view.nTracks(); idx < nt; idx += gridDim.x * blockDim.x) {
      auto nHits = helper::nHits(tracks_view, idx);
      assert(nHits >= 3);

      // initialize soa...
      soa[idx].idv() = -1;

      if (helper::isTriplet(tracks_view, idx))
        continue;  // no triplets
      if (quality[idx] < pixelTrack::Quality::highPurity)
        continue;

      auto pt = tracks_view[idx].pt();

      if (pt < ptMin)
        continue;

      // clamp pt
      pt = std::min(pt, ptMax);

      auto& data = pws;
      auto it = atomicAdd(&data.ntrks(), 1);
      data[it].itrk() = idx;
      data[it].zt() = helper::zip(tracks_view, idx);
      data[it].ezt2() = tracks_view[idx].covariance()(14);
      data[it].ptt2() = pt * pt;
    }
  }

// #define THREE_KERNELS
#ifndef THREE_KERNELS
  __global__ void vertexFinderOneKernel(VtxSoAView pdata,
                                        WsSoAView pws,
                                        int minT,      // min number of neighbours to be "seed"
                                        float eps,     // max absolute distance to cluster
                                        float errmax,  // max error to be "seed"
                                        float chi2max  // max normalized distance to cluster,
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
    __syncthreads();
    fitVertices(pdata, pws, maxChi2ForFirstFit);
    __syncthreads();
    splitVertices(pdata, pws, maxChi2ForSplit);
    __syncthreads();
    fitVertices(pdata, pws, maxChi2ForFinalFit);
    __syncthreads();
    sortByPt2(pdata, pws);
  }
#else
  __global__ void vertexFinderKernel1(VtxSoAView pdata,
                                      WsSoAView pws,
                                      int minT,      // min number of neighbours to be "seed"
                                      float eps,     // max absolute distance to cluster
                                      float errmax,  // max error to be "seed"
                                      float chi2max  // max normalized distance to cluster,
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
    __syncthreads();
    fitVertices(pdata, pws, maxChi2ForFirstFit);
  }

  __global__ void vertexFinderKernel2(VtxSoAView pdata, WsSoAView pws) {
    fitVertices(pdata, pws, maxChi2ForFinalFit);
    __syncthreads();
    sortByPt2(pdata, pws);
  }
#endif

  template <typename TrackerTraits>
#ifdef __CUDACC__
  ZVertexSoADevice Producer<TrackerTraits>::makeAsync(cudaStream_t stream,
                                                      const TrackSoAConstView<TrackerTraits>& tracks_view,
                                                      float ptMin,
                                                      float ptMax) const {
#ifdef PIXVERTEX_DEBUG_PRODUCE
    std::cout << "producing Vertices on GPU" << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
    ZVertexSoADevice vertices(stream);
#else
  ZVertexSoAHost Producer<TrackerTraits>::make(const TrackSoAConstView<TrackerTraits>& tracks_view,
                                               float ptMin,
                                               float ptMax) const {
#ifdef PIXVERTEX_DEBUG_PRODUCE
    std::cout << "producing Vertices on  CPU" << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
    ZVertexSoAHost vertices;
#endif
    auto soa = vertices.view();

    assert(vertices.buffer());

#ifdef __CUDACC__
    auto ws_d = gpuVertexFinder::workSpace::PixelVertexWorkSpaceSoADevice(stream);
#else
    auto ws_d = gpuVertexFinder::workSpace::PixelVertexWorkSpaceSoAHost();
#endif

#ifdef __CUDACC__
    init<<<1, 1, 0, stream>>>(soa, ws_d.view());
    auto blockSize = 128;
    auto numberOfBlocks = (tracks_view.metadata().size() + blockSize - 1) / blockSize;
    loadTracks<TrackerTraits><<<numberOfBlocks, blockSize, 0, stream>>>(tracks_view, soa, ws_d.view(), ptMin, ptMax);
    cudaCheck(cudaGetLastError());
#else
    init(soa, ws_d.view());
    loadTracks<TrackerTraits>(tracks_view, soa, ws_d.view(), ptMin, ptMax);
#endif

#ifdef __CUDACC__
    // Running too many thread lead to problems when printf is enabled.
    constexpr int maxThreadsForPrint = 1024 - 128;
    constexpr int numBlocks = 1024;
    constexpr int threadsPerBlock = 128;

    if (oneKernel_) {
      // implemented only for density clustesrs
#ifndef THREE_KERNELS
      vertexFinderOneKernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view(), minT, eps, errmax, chi2max);
#else
      vertexFinderKernel1<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view(), minT, eps, errmax, chi2max);
      cudaCheck(cudaGetLastError());
      // one block per vertex...
      splitVerticesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(soa, ws_d.view(), maxChi2ForSplit);
      cudaCheck(cudaGetLastError());
      vertexFinderKernel2<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view());
#endif
    } else {  // five kernels
      if (useDensity_) {
        clusterTracksByDensityKernel<<<1, maxThreadsForPrint, 0, stream>>>(
            soa, ws_d.view(), minT, eps, errmax, chi2max);
      } else if (useDBSCAN_) {
        clusterTracksDBSCAN<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view(), minT, eps, errmax, chi2max);
      } else if (useIterative_) {
        clusterTracksIterative<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view(), minT, eps, errmax, chi2max);
      }
      cudaCheck(cudaGetLastError());
      fitVerticesKernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view(), maxChi2ForFirstFit);
      cudaCheck(cudaGetLastError());
      // one block per vertex...
      splitVerticesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(soa, ws_d.view(), maxChi2ForSplit);
      cudaCheck(cudaGetLastError());
      fitVerticesKernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view(), maxChi2ForFinalFit);
      cudaCheck(cudaGetLastError());
      sortByPt2Kernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.view());
    }
    cudaCheck(cudaGetLastError());
#else  // __CUDACC__
    if (useDensity_) {
      clusterTracksByDensity(soa, ws_d.view(), minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN(soa, ws_d.view(), minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative(soa, ws_d.view(), minT, eps, errmax, chi2max);
    }
#ifdef PIXVERTEX_DEBUG_PRODUCE
    std::cout << "found " << ws_d.view().nvIntermediate() << " vertices " << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
    fitVertices(soa, ws_d.view(), maxChi2ForFirstFit);
    // one block per vertex!
    splitVertices(soa, ws_d.view(), maxChi2ForSplit);
    fitVertices(soa, ws_d.view(), maxChi2ForFinalFit);
    sortByPt2(soa, ws_d.view());
#endif

    return vertices;
  }

  template class Producer<pixelTopology::Phase1>;
  template class Producer<pixelTopology::Phase2>;
}  // namespace gpuVertexFinder
