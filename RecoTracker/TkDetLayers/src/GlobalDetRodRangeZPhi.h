#ifndef RecoTracker_TkDetLayers_GlobalDetRodRangeZPhi_h
#define RecoTracker_TkDetLayers_GlobalDetRodRangeZPhi_h

#include <utility>

class Plane;

/** Implementation class for PhiZMeasurementEstimator etc.
 */

#pragma GCC visibility push(hidden)
class GlobalDetRodRangeZPhi {
public:

  typedef std::pair<float,float> Range;

  GlobalDetRodRangeZPhi( const Plane& RodPlane);

  Range zRange() const { return theZRange;}
  Range phiRange() const { return thePhiRange;}

private:
  Range theZRange;
  Range thePhiRange;
};

#pragma GCC visibility pop
#endif
