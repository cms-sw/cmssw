#include <alpaka/alpaka.hpp>
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/PixelVertexWorkSpaceSoADeviceAlpaka.h"

#include "vertexFinder.h"
#include "vertexFinder.h"
#include "clusterTracksDBSCAN.h"
#include "clusterTracksIterative.h"
#include "clusterTracksByDensity.h"
#include "fitVertices.h"
#include "sortByPt2.h"
#include "splitVertices.h"

#undef PIXVERTEX_DEBUG_PRODUCE
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace vertexFinder {
    using namespace cms::alpakatools;
    // reject outlier tracks that contribute more than this to the chi2 of the vertex fit
    constexpr float maxChi2ForFirstFit = 50.f;
    constexpr float maxChi2ForFinalFit = 5000.f;

    // split vertices with a chi2/NDoF greater than this
    constexpr float maxChi2ForSplit = 9.f;

    template <typename TrackerTraits>
    class LoadTracks {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    reco::TrackSoAConstView<TrackerTraits> tracks_view,
                                    VtxSoAView soa,
                                    WsSoAView pws,
                                    float ptMin,
                                    float ptMax) const {
        auto const* quality = tracks_view.quality();
        using helper = TracksUtilities<TrackerTraits>;

        for (auto idx : cms::alpakatools::elements_with_stride(acc, tracks_view.nTracks())) {
          [[maybe_unused]] auto nHits = helper::nHits(tracks_view, idx);
          ALPAKA_ASSERT_OFFLOAD(nHits >= 3);

          // initialize soa...
          soa[idx].idv() = -1;

          if (reco::isTriplet(tracks_view, idx))
            continue;  // no triplets
          if (quality[idx] < ::pixelTrack::Quality::highPurity)
            continue;

          auto pt = tracks_view[idx].pt();

          if (pt < ptMin)
            continue;

          // clamp pt
          pt = std::min<float>(pt, ptMax);

          auto& data = pws;
          auto it = alpaka::atomicAdd(acc, &data.ntrks(), 1u, alpaka::hierarchy::Blocks{});
          data[it].itrk() = idx;
          data[it].zt() = reco::zip(tracks_view, idx);
          data[it].ezt2() = tracks_view[idx].covariance()(14);
          data[it].ptt2() = pt * pt;
        }
      }
    };
// #define THREE_KERNELS
#ifndef THREE_KERNELS
    class VertexFinderOneKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VtxSoAView pdata,
                                    WsSoAView pws,
                                    bool doSplit,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster,
      ) const {
        clusterTracksByDensity(acc, pdata, pws, minT, eps, errmax, chi2max);
        alpaka::syncBlockThreads(acc);
        fitVertices(acc, pdata, pws, maxChi2ForFirstFit);
        alpaka::syncBlockThreads(acc);
        if (doSplit) {
          splitVertices(acc, pdata, pws, maxChi2ForSplit);
          alpaka::syncBlockThreads(acc);
          fitVertices(acc, pdata, pws, maxChi2ForFinalFit);
          alpaka::syncBlockThreads(acc);
        }
        sortByPt2(acc, pdata, pws);
      }
    };
#else
    class VertexFinderKernel1 {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VtxSoAView pdata,
                                    WsSoAView pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster,
      ) const {
        clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
        alpaka::syncBlockThreads(acc);
        fitVertices(pdata, pws, maxChi2ForFirstFit);
      }
    };
    class VertexFinderKernel2 {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, VtxSoAView pdata, WsSoAView pws) const {
        fitVertices(pdata, pws, maxChi2ForFinalFit);
        alpaka::syncBlockThreads(acc);
        sortByPt2(pdata, pws);
      }
    };
#endif

    template <typename TrackerTraits>
    ZVertexSoACollection Producer<TrackerTraits>::makeAsync(Queue& queue,
                                                            const reco::TrackSoAConstView<TrackerTraits>& tracks_view,
                                                            float ptMin,
                                                            float ptMax) const {
#ifdef PIXVERTEX_DEBUG_PRODUCE
      std::cout << "producing Vertices on GPU" << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
      ZVertexSoACollection vertices(queue);

      auto soa = vertices.view();

      auto ws_d = PixelVertexWorkSpaceSoADevice(::zVertex::MAXTRACKS, queue);

      // Initialize
      const auto initWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue, initWorkDiv, Init{}, soa, ws_d.view());

      // Load Tracks
      const uint32_t blockSize = 128;
      const uint32_t numberOfBlocks =
          cms::alpakatools::divide_up_by(tracks_view.metadata().size() + blockSize - 1, blockSize);
      const auto loadTracksWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(
          queue, loadTracksWorkDiv, LoadTracks<TrackerTraits>{}, tracks_view, soa, ws_d.view(), ptMin, ptMax);

      // Running too many thread lead to problems when printf is enabled.
      const auto finderSorterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024 - 128);
      const auto splitterFitterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1024, 128);

      if (oneKernel_) {
        // implemented only for density clustesrs
#ifndef THREE_KERNELS
        alpaka::exec<Acc1D>(queue,
                            finderSorterWorkDiv,
                            VertexFinderOneKernel{},
                            soa,
                            ws_d.view(),
                            doSplitting_,
                            minT,
                            eps,
                            errmax,
                            chi2max);
#else
        alpaka::exec<Acc1D>(
            queue, finderSorterWorkDiv, VertexFinderOneKernel{}, soa, ws_d.view(), minT, eps, errmax, chi2max);

        // one block per vertex...
        if (doSplitting_)
          alpaka::exec<Acc1D>(queue, splitterFitterWorkDiv, SplitVerticesKernel{}, soa, ws_d.view(), maxChi2ForSplit);
        alpaka::exec<Acc1D>(queue, finderSorterWorkDiv{}, soa, ws_d.view());
#endif
      } else {  // five kernels
        if (useDensity_) {
          alpaka::exec<Acc1D>(
              queue, finderSorterWorkDiv, ClusterTracksByDensityKernel{}, soa, ws_d.view(), minT, eps, errmax, chi2max);

        } else if (useDBSCAN_) {
          alpaka::exec<Acc1D>(
              queue, finderSorterWorkDiv, ClusterTracksDBSCAN{}, soa, ws_d.view(), minT, eps, errmax, chi2max);
        } else if (useIterative_) {
          alpaka::exec<Acc1D>(
              queue, finderSorterWorkDiv, ClusterTracksIterative{}, soa, ws_d.view(), minT, eps, errmax, chi2max);
        }
        alpaka::exec<Acc1D>(queue, finderSorterWorkDiv, FitVerticesKernel{}, soa, ws_d.view(), maxChi2ForFirstFit);

        // one block per vertex...
        if (doSplitting_) {
          alpaka::exec<Acc1D>(queue, splitterFitterWorkDiv, SplitVerticesKernel{}, soa, ws_d.view(), maxChi2ForSplit);

          alpaka::exec<Acc1D>(queue, finderSorterWorkDiv, FitVerticesKernel{}, soa, ws_d.view(), maxChi2ForFinalFit);
        }
        alpaka::exec<Acc1D>(queue, finderSorterWorkDiv, SortByPt2Kernel{}, soa, ws_d.view());
      }

      return vertices;
    }

    template class Producer<pixelTopology::Phase1>;
    template class Producer<pixelTopology::Phase2>;
    template class Producer<pixelTopology::HIonPhase1>;
  }  // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
