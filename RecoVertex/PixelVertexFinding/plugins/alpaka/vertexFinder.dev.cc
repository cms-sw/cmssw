#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoVertex/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoVertex/PixelVertexFinding/plugins/alpaka/PixelVertexWorkSpaceSoADeviceAlpaka.h"

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
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    reco::TrackSoAConstView<TrackerTraits> tracks_view,
                                    VtxSoAView data,
                                    TrkSoAView trkdata,
                                    WsSoAView ws,
                                    float ptMin,
                                    float ptMax) const {
        auto const* quality = tracks_view.quality();
        using helper = TracksUtilities<TrackerTraits>;

        for (auto idx : cms::alpakatools::uniform_elements(acc, tracks_view.nTracks())) {
          [[maybe_unused]] auto nHits = helper::nHits(tracks_view, idx);
          ALPAKA_ASSERT_ACC(nHits >= 3);

          // initialize the track data
          trkdata[idx].idv() = -1;

          // do not use triplets
          if (reco::isTriplet(tracks_view, idx))
            continue;

          // use only "high purity" track
          if (quality[idx] < ::pixelTrack::Quality::highPurity)
            continue;

          auto pt = tracks_view[idx].pt();
          // pT min cut
          if (pt < ptMin)
            continue;

          // clamp pT to the pTmax
          pt = std::min<float>(pt, ptMax);

          // load the track data into the workspace
          auto it = alpaka::atomicAdd(acc, &ws.ntrks(), 1u, alpaka::hierarchy::Blocks{});
          ws[it].itrk() = idx;
          ws[it].zt() = reco::zip(tracks_view, idx);
          ws[it].ezt2() = tracks_view[idx].covariance()(14);
          ws[it].ptt2() = pt * pt;
        }
      }
    };

// #define THREE_KERNELS
#ifndef THREE_KERNELS
    class VertexFinderOneKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    VtxSoAView data,
                                    TrkSoAView trkdata,
                                    WsSoAView ws,
                                    bool doSplit,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster,
      ) const {
        clusterTracksByDensity(acc, data, trkdata, ws, minT, eps, errmax, chi2max);
        alpaka::syncBlockThreads(acc);
        fitVertices(acc, data, trkdata, ws, maxChi2ForFirstFit);
        alpaka::syncBlockThreads(acc);
        if (doSplit) {
          splitVertices(acc, data, trkdata, ws, maxChi2ForSplit);
          alpaka::syncBlockThreads(acc);
          fitVertices(acc, data, trkdata, ws, maxChi2ForFinalFit);
          alpaka::syncBlockThreads(acc);
        }
        sortByPt2(acc, data, trkdata, ws);
      }
    };
#else
    class VertexFinderKernel1 {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    VtxSoAView data,
                                    WsSoAView ws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster,
      ) const {
        clusterTracksByDensity(acc, data, ws, minT, eps, errmax, chi2max);
        alpaka::syncBlockThreads(acc);
        fitVertices(acc, data, ws, maxChi2ForFirstFit);
      }
    };

    class VertexFinderKernel2 {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, VtxSoAView data, WsSoAView ws) const {
        fitVertices(acc, data, ws, maxChi2ForFinalFit);
        alpaka::syncBlockThreads(acc);
        sortByPt2(data, ws);
      }
    };
#endif

    template <typename TrackerTraits>
    ZVertexSoACollection Producer<TrackerTraits>::makeAsync(Queue& queue,
                                                            reco::TrackSoAConstView<TrackerTraits> const& tracks_view,
                                                            int maxVertices,
                                                            float ptMin,
                                                            float ptMax) const {
#ifdef PIXVERTEX_DEBUG_PRODUCE
      std::cout << "producing Vertices on GPU" << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
      const auto maxTracks = tracks_view.metadata().size();
      ZVertexSoACollection vertices({{maxVertices, maxTracks}}, queue);
      auto data = vertices.view();
      auto trkdata = vertices.view<reco::ZVertexTracksSoA>();

      PixelVertexWorkSpaceSoADevice workspace(maxTracks, queue);
      auto ws = workspace.view();

      // Initialize
      const auto initWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue, initWorkDiv, Init{}, data, ws);

      // Load Tracks
      const uint32_t blockSize = 128;
      const uint32_t numberOfBlocks =
          cms::alpakatools::divide_up_by(tracks_view.metadata().size() + blockSize - 1, blockSize);
      const auto loadTracksWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(
          queue, loadTracksWorkDiv, LoadTracks<TrackerTraits>{}, tracks_view, data, trkdata, ws, ptMin, ptMax);

      // Running too many thread lead to problems when printf is enabled.
      const auto finderSorterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024 - 128);
      const auto splitterFitterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1024, 128);

      if (oneKernel_) {
        // implemented only for density clustesrs
#ifndef THREE_KERNELS
        alpaka::exec<Acc1D>(queue,
                            finderSorterWorkDiv,
                            VertexFinderOneKernel{},
                            data,
                            trkdata,
                            ws,
                            doSplitting_,
                            minT,
                            eps,
                            errmax,
                            chi2max);
#else
        alpaka::exec<Acc1D>(
            queue, finderSorterWorkDiv, VertexFinderOneKernel{}, data, trkdata, ws, minT, eps, errmax, chi2max);

        // one block per vertex...
        if (doSplitting_)
          alpaka::exec<Acc1D>(queue, splitterFitterWorkDiv, SplitVerticesKernel{}, data, trkdata, ws, maxChi2ForSplit);
        alpaka::exec<Acc1D>(queue, finderSorterWorkDiv{}, data, ws);
#endif
      } else {  // five kernels
        if (useDensity_) {
          alpaka::exec<Acc1D>(
              queue, finderSorterWorkDiv, ClusterTracksByDensityKernel{}, data, trkdata, ws, minT, eps, errmax, chi2max);

        } else if (useDBSCAN_) {
          alpaka::exec<Acc1D>(
              queue, finderSorterWorkDiv, ClusterTracksDBSCAN{}, data, trkdata, ws, minT, eps, errmax, chi2max);
        } else if (useIterative_) {
          alpaka::exec<Acc1D>(
              queue, finderSorterWorkDiv, ClusterTracksIterative{}, data, trkdata, ws, minT, eps, errmax, chi2max);
        }
        alpaka::exec<Acc1D>(queue, finderSorterWorkDiv, FitVerticesKernel{}, data, trkdata, ws, maxChi2ForFirstFit);

        // one block per vertex...
        if (doSplitting_) {
          alpaka::exec<Acc1D>(queue, splitterFitterWorkDiv, SplitVerticesKernel{}, data, trkdata, ws, maxChi2ForSplit);

          alpaka::exec<Acc1D>(queue, finderSorterWorkDiv, FitVerticesKernel{}, data, trkdata, ws, maxChi2ForFinalFit);
        }
        alpaka::exec<Acc1D>(queue, finderSorterWorkDiv, SortByPt2Kernel{}, data, trkdata, ws);
      }

      return vertices;
    }

    template class Producer<pixelTopology::Phase1>;
    template class Producer<pixelTopology::Phase2>;
    template class Producer<pixelTopology::HIonPhase1>;
    template class Producer<pixelTopology::Phase1Strip>;
  }  // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
