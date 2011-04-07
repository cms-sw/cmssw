#ifndef RecoTracker_TkDetLayers_GlobalDetRodRangeZPhi_h
#define RecoTracker_TkDetLayers_GlobalDetRodRangeZPhi_h

#include <utility>

class BoundPlane;

/** Implementation class for PhiZMeasurementEstimator etc.
 */

class GlobalDetRodRangeZPhi {
public:

  typedef std::pair<float,float> Range;

  GlobalDetRodRangeZPhi( const BoundPlane& RodPlane);

  Range zRange() const { return theZRange;}
  Range phiRange() const { return thePhiRange;}

private:
  Range theZRange;
  Range thePhiRange;
};

#endif
