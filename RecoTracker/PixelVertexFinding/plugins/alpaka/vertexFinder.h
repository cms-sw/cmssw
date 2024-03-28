#ifndef RecoTracker_PixelVertexFinding_plugins_alpaka_vertexFinder_h
#define RecoTracker_PixelVertexFinding_plugins_alpaka_vertexFinder_h

#include <cstddef>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/PixelVertexWorkSpaceSoADeviceAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder {

  using namespace cms::alpakatools;
  using VtxSoAView = ::reco::ZVertexSoAView;
  using TrkSoAView = ::reco::ZVertexTracksSoAView;
  using WsSoAView = ::vertexFinder::PixelVertexWorkSpaceSoAView;

  class Init {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, VtxSoAView pdata, WsSoAView pws) const {
      pdata.nvFinal() = 0;  // initialization
      ::vertexFinder::init(pws);
    }
  };

  template <typename TrackerTraits>
  class Producer {
    using TkSoAConstView = reco::TrackSoAConstView<TrackerTraits>;

  public:
    Producer(bool oneKernel,
             bool useDensity,
             bool useDBSCAN,
             bool useIterative,
             bool doSplitting,
             int iminT,      // min number of neighbours to be "core"
             float ieps,     // max absolute distance to cluster
             float ierrmax,  // max error to be "seed"
             float ichi2max  // max normalized distance to cluster
             )
        : oneKernel_(oneKernel && !(useDBSCAN || useIterative)),
          useDensity_(useDensity),
          useDBSCAN_(useDBSCAN),
          useIterative_(useIterative),
          doSplitting_(doSplitting),
          minT(iminT),
          eps(ieps),
          errmax(ierrmax),
          chi2max(ichi2max) {}

    ~Producer() = default;

    ZVertexSoACollection makeAsync(Queue &queue, const TkSoAConstView &tracks_view, float ptMin, float ptMax) const;

  private:
    const bool oneKernel_;     // run everything (cluster,fit,split,sort) in one kernel. Uses only density clusterizer
    const bool useDensity_;    // use density clusterizer
    const bool useDBSCAN_;     // use DBScan clusterizer
    const bool useIterative_;  // use iterative clusterizer
    const bool doSplitting_;   //run vertex splitting

    int minT;       // min number of neighbours to be "core"
    float eps;      // max absolute distance to cluster
    float errmax;   // max error to be "seed"
    float chi2max;  // max normalized distance to cluster
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder

#endif  // RecoTracker_PixelVertexFinding_plugins_alpaka_vertexFinder_h
