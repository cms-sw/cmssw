#ifndef PixelFitterByConformalMappingAndLine_H
#define PixelFitterByConformalMappingAndLine_H

#include "RecoTracker/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/TrackReco/interface/Track.h"

class TrackerGeometry;
class MagneticField;
class TransientTrackingRecHitBuilder;

class PixelFitterByConformalMappingAndLine : public PixelFitterBase {
public:
  explicit PixelFitterByConformalMappingAndLine(const TransientTrackingRecHitBuilder *ttrhBuilder,
                                                const TrackerGeometry *tracker,
                                                const MagneticField *field,
                                                double fixImpactParameter,
                                                bool useFixImpactParameter);
  ~PixelFitterByConformalMappingAndLine() override {}
  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *> &hits,
                                   const TrackingRegion &region) const override;

private:
  const TransientTrackingRecHitBuilder *theTTRHBuilder;
  const TrackerGeometry *theTracker;
  const MagneticField *theField;
  const double theFixImpactParameter;
  const bool theUseFixImpactParameter;
};
#endif
