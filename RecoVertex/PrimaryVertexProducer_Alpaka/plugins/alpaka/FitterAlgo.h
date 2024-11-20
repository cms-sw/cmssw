#ifndef RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h
#define RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h

#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct fitterParameters {
    double chi2cutoff;  // Unused?
    double minNdof;     // Unused?
    bool useBeamSpotConstraint;
    double maxDistanceToBeam;  // Unused?
  };

  class FitterAlgo {
  public:
    FitterAlgo(Queue& queue, const int32_t nV, fitterParameters fPar);  // Just configuration and making job divisions
    void fit(Queue& queue,
             const portablevertex::TrackDeviceCollection& deviceTrack,
             portablevertex::VertexDeviceCollection& deviceVertex,
             const portablevertex::BeamSpotDeviceCollection& deviceBeamSpot);  // The actual fitting
  private:
    cms::alpakatools::device_buffer<Device, bool> useBeamSpotConstraint;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoVertex_PrimaryVertexProducer_Alpaka_plugins_alpaka_FitterAlgo_h
