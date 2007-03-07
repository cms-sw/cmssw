#ifndef CloseComponentsMerger_H
#define CloseComponentsMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include <map>


/** Merging of a Gaussian mixture by clustering components
 *  which are close to one another. The actual calculation
 *  of the distance between components is done by a specific
 *  (polymorphic) class, given at construction time.
 */

class CloseComponentsMerger : public MultiGaussianStateMerger {

 public:

  CloseComponentsMerger(int n,
			const DistanceBetweenComponents* distance);

  virtual CloseComponentsMerger* clone() const
  {  
    return new CloseComponentsMerger(*this);
  }
  
  /** Method which does the actual merging. Returns a trimmed MultiGaussianState.
   */

  virtual RCMultiGaussianState merge(const RCMultiGaussianState& mgs) const;
  
 private:
  
  typedef RCSingleGaussianState SGS;
  typedef std::multimap<double, RCSingleGaussianState> SingleStateMap;

  std::pair<SGS, SingleStateMap::iterator> compWithMinDistToLargestWeight(SingleStateMap&) const;

  int theMaxNumberOfComponents;
  DeepCopyPointerByClone<DistanceBetweenComponents> theDistance;

};  

#endif // CloseComponentsMerger_H
