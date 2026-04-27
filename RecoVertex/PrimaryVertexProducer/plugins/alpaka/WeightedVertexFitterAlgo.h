#ifndef RecoVertex_PrimaryVertexProducer_plugins_alpaka_WeightedVertexFitterAlgo_h
#define RecoVertex_PrimaryVertexProducer_plugins_alpaka_WeightedVertexFitterAlgo_h

#include "DataFormats/BeamSpot/interface/alpaka/BeamSpotDevice.h"
#include "DataFormats/VertexSoA/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/TrackForVertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct FitterParameters {
    // In principle a single one, but in case we end up wanting to add more configuration to the fiter
    bool useBeamSpotConstraint;
  };

  class WeightedVertexFitterAlgo {
  public:
    WeightedVertexFitterAlgo(Queue& queue, FitterParameters fPar);
    void fit(Queue& queue,
             const TrackForVertexDeviceCollection& deviceTrack,
             VertexDeviceCollection& deviceVertex,
             const BeamSpotDevice& deviceBeamSpot);

  private:
    cms::alpakatools::device_buffer<Device, bool> useBeamSpotConstraint;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_plugins_alpaka_WeightedVertexFitterAlgo_h
