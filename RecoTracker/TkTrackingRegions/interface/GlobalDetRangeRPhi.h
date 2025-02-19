#ifndef RecoTracker_TkTrackingRegions_GlobalDetRangeRPhi_H
#define RecoTracker_TkTrackingRegions_GlobalDetRangeRPhi_H

#include <utility>

class BoundPlane;

/** Keep R and Phi range for detunit */

class GlobalDetRangeRPhi {
public:

  typedef std::pair<float,float> Range;

  GlobalDetRangeRPhi( const BoundPlane& det);

  Range rRange() const { return theRRange;}
  Range phiRange() const { return thePhiRange;}

private:
  Range theRRange;
  Range thePhiRange;
};

#endif
