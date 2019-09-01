#ifndef CloseComponentsMerger_H
#define CloseComponentsMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include <map>
#include <algorithm>

/** Merging of a Gaussian mixture by clustering components
 *  which are close to one another. The actual calculation
 *  of the distance between components is done by a specific
 *  (polymorphic) class, given at construction time.
 */

template <unsigned int N>
class CloseComponentsMerger final : public MultiGaussianStateMerger<N> {
private:
  using SingleState = SingleGaussianState<N>;
  using MultiState = MultiGaussianState<N>;
  using SingleStatePtr = std::shared_ptr<SingleState>;

public:
  CloseComponentsMerger(int n, const DistanceBetweenComponents<N>* distance);

  CloseComponentsMerger* clone() const override { return new CloseComponentsMerger(*this); }

  /** Method which does the actual merging. Returns a trimmed MultiGaussianState.
   */

  MultiState merge(const MultiState& mgs) const override;

  MultiState mergeOld(const MultiState& mgs) const;

public:
  typedef std::multimap<double, SingleStatePtr> SingleStateMap;
  typedef std::pair<SingleStatePtr, typename SingleStateMap::iterator> MinDistResult;

private:
  //   std::pair< SingleState, SingleStateMap::iterator >
  MinDistResult compWithMinDistToLargestWeight(SingleStateMap&) const;

  int theMaxNumberOfComponents;
  DeepCopyPointerByClone<DistanceBetweenComponents<N> > theDistance;
};

#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.icc"

#endif  // CloseComponentsMerger_H
