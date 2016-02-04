#ifndef CloseComponentsMerger_H
#define CloseComponentsMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "boost/shared_ptr.hpp"
#include <map>


/** Merging of a Gaussian mixture by clustering components
 *  which are close to one another. The actual calculation
 *  of the distance between components is done by a specific
 *  (polymorphic) class, given at construction time.
 */

template <unsigned int N>
class CloseComponentsMerger : public MultiGaussianStateMerger<N> {

 private:
  typedef SingleGaussianState<N> SingleState;
  typedef MultiGaussianState<N> MultiState;
  typedef boost::shared_ptr<SingleState> SingleStatePtr;

 public:

  CloseComponentsMerger(int n,
			   const DistanceBetweenComponents<N>* distance);

  virtual CloseComponentsMerger* clone() const
  {  
    return new CloseComponentsMerger(*this);
  }
  
  /** Method which does the actual merging. Returns a trimmed MultiGaussianState.
   */

  virtual MultiState merge(const MultiState& mgs) const;
  

public:
  typedef std::multimap< double, SingleStatePtr > SingleStateMap;
  typedef std::pair< SingleStatePtr, typename SingleStateMap::iterator > MinDistResult;

private:

//   std::pair< SingleState, SingleStateMap::iterator > 
  MinDistResult
  compWithMinDistToLargestWeight(SingleStateMap&) const;

  int theMaxNumberOfComponents;
  DeepCopyPointerByClone< DistanceBetweenComponents<N> > theDistance;

};  

#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.icc"

#endif // CloseComponentsMerger_H
