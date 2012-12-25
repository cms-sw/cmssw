#ifndef RecoTracker_TkTrackingRegions_GlobalDetRangeZPhi_H
#define RecoTracker_TkTrackingRegions_GlobalDetRangeZPhi_H

#include <utility>

class Plane;

/** Implementation class for PhiZMeasurementEstimator etc.
 */

class GlobalDetRangeZPhi {
public:

  typedef std::pair<float,float> Range;

  GlobalDetRangeZPhi( const Plane& det);

  Range zRange() const { return theZRange;}
  Range phiRange() const { return thePhiRange;}

private:
  Range theZRange;
  Range thePhiRange;
};

#endif
