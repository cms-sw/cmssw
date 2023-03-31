#ifndef TrackFitter_H
#define TrackFitter_H

#include "RecoTracker/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class TransientTrackingRecHitBuilder;
class TrackerGeometry;
class MagneticField;

class TrackFitter : public PixelFitterBase {
public:
  TrackFitter(const TrackerGeometry* tracker,
              const MagneticField* field,
              const TransientTrackingRecHitBuilder* ttrhBuilder)
      : theTracker(tracker), theField(field), theTTRecHitBuilder(ttrhBuilder) {}
  ~TrackFitter() override {}

  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit*>& hits,
                                   const TrackingRegion& region) const override;

private:
  float getCotThetaAndUpdateZip(
      const GlobalPoint& inner, const GlobalPoint& outer, float radius, float phi, float d0, float& zip) const;
  float getPhi(float xC, float yC, int charge) const;
  float getZip(float d0, float curv, const GlobalPoint& inner, const GlobalPoint& outer) const;
  void getErrTipAndErrZip(float pt, float eta, float& errZip, float& errTip) const;

  const TrackerGeometry* theTracker;
  const MagneticField* theField;
  const TransientTrackingRecHitBuilder* theTTRecHitBuilder;
};
#endif
