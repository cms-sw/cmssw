#ifndef RecoTracker_PixelTrackFitting_PixelFitterByHelixProjections_h
#define RecoTracker_PixelTrackFitting_PixelFitterByHelixProjections_h

#include "RecoTracker/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class TrackerTopology;

class PixelFitterByHelixProjections final : public PixelFitterBase {
public:
  explicit PixelFitterByHelixProjections(const TrackerTopology *ttopo,
                                         const MagneticField *field,
                                         bool scaleErrorsForBPix1,
                                         float scaleFactor);
  ~PixelFitterByHelixProjections() override {}
  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *> &hits,
                                   const TrackingRegion &region) const override;

private:
  const TrackerTopology *theTopo;
  const MagneticField *theField;
  const bool thescaleErrorsForBPix1;
  const float thescaleFactor;
};
#endif
