#ifndef RecoPixelVertexing_PixelTrackFitting_interface_PixelNtupletsFitter_h
#define RecoPixelVertexing_PixelTrackFitting_interface_PixelNtupletsFitter_h

#include <vector>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelFitterBase.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PixelNtupletsFitter final : public PixelFitterBase {
  public:
    explicit PixelNtupletsFitter(Queue& queue, float nominalB, const MagneticField* field, bool useRiemannFit);
    ~PixelNtupletsFitter() override = default;
    std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit*>& hits,
                                     const TrackingRegion& region) const override;

  private:
    Queue& queue_;
    float nominalB_;
    const MagneticField* field_;
    bool useRiemannFit_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoPixelVertexing_PixelTrackFitting_interface_PixelNtupletsFitter_h
