#ifndef RecoTracker_PixelLowPtUtilities_ClusterData_h
#define RecoTracker_PixelLowPtUtilities_ClusterData_h

#include "FWCore/Utilities/interface/VecArray.h"
#include <utility>

class ClusterData {
public:
  using ArrayType = edm::VecArray<std::pair<int, int>, 9>;
  ArrayType size;
  bool isStraight, isComplete, hasBigPixelsOnlyInside;
};

#endif
