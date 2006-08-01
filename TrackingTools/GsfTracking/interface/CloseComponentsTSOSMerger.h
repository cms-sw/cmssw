#ifndef CloseComponentsTSOSMerger_H
#define CloseComponentsTSOSMerger_H

#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "TrackingTools/GsfTracking/interface/TSOSDistanceBetweenComponents.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include <map>

/** Merging of a Gaussian mixture by clustering components
 *  which are close to one another. The actual calculation
 *  of the distance between components is done by a specific
 *  (polymorphic) class, given at construction time.
 */

class CloseComponentsTSOSMerger : public MultiTrajectoryStateMerger 
{

 public:
  CloseComponentsTSOSMerger (int maxNumberOfComponents,
			     const TSOSDistanceBetweenComponents* distance);

  virtual CloseComponentsTSOSMerger* clone() const
  {  
    return new CloseComponentsTSOSMerger(*this);
  }
  
  /// Method which does the actual merging. Returns a trimmed TSOS.
  virtual TrajectoryStateOnSurface merge(const TrajectoryStateOnSurface& tsos) const;
  
 private:
  typedef TrajectoryStateOnSurface TSOS;
  typedef std::multimap<double, TSOS> TsosMap;

  std::pair<TSOS, TsosMap::iterator>  compWithMinDistToLargestWeight(TsosMap&) const;

  int theMaxNumberOfComponents;
  DeepCopyPointerByClone<TSOSDistanceBetweenComponents> theDistance;
};  

#endif // CloseComponentsTSOSMerger_H
