#ifndef LargestWeightsTSOSMerger_H
#define LargestWeightsTSOSMerger_H

#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"

class TrajectoryStateOnSurface;

/** Merging of a Gaussian mixture by keeping
 *  the N components with the largest weights.
 */

class LargestWeightsTSOSMerger : public MultiTrajectoryStateMerger
{

public:
  LargestWeightsTSOSMerger (int maxNrOfComponents,
			    bool smallestWeightsMerging);
  
  
  virtual LargestWeightsTSOSMerger* clone() const
  {  
    return new LargestWeightsTSOSMerger(*this);
  }
  
  /// Method which does the actual merging. Returns a trimmed TSOS.
  virtual TrajectoryStateOnSurface merge(const TrajectoryStateOnSurface& tsos) const;
  
 private:
  unsigned int theMaxNumberOfComponents;
  bool theSmallestWeightsMerging;
};  

#endif
