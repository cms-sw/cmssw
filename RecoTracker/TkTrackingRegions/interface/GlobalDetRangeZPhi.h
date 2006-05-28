#ifndef RecoTracker_TkTrackingRegions_GlobalDetRangeZPhi_H
#define RecoTracker_TkTrackingRegions_GlobalDetRangeZPhi_H

#include <utility>
using namespace std;
class BoundPlane;

/** Implementation class for PhiZMeasurementEstimator etc.
 */

class GlobalDetRangeZPhi {
public:

  typedef pair<float,float> Range;

  GlobalDetRangeZPhi( const BoundPlane& det);

  Range zRange() const { return theZRange;}
  Range phiRange() const { return thePhiRange;}

private:
  Range theZRange;
  Range thePhiRange;
};

#endif
