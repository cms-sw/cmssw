#ifndef RecoTracker_PixelTrackFitting_PixelTrackBuilder_h
#define RecoTracker_PixelTrackFitting_PixelTrackBuilder_h

#include <vector>
#include <string>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
class TrackingRecHit;
class MagneticField;
class FreeTrajectoryState;

class PixelTrackBuilder {
public:
  reco::Track* build(const Measurement1D& pt,        // transverse momentu
                     const Measurement1D& phi,       // direction at impact point
                     const Measurement1D& cotTheta,  // cotangent of polar angle
                     const Measurement1D& tip,       // closest approach in 2D
                     const Measurement1D& zip,       // z at closest approach in 2D
                     float chi2,                     // chi2
                     int charge,                     // chi2
                     const std::vector<const TrackingRecHit*>& hits,
                     const MagneticField* mf,
                     // reference point of a track for IP computation
                     const GlobalPoint& reference = GlobalPoint(0, 0, 0)) const;
};

#endif
