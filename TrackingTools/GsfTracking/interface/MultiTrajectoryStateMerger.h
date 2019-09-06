#ifndef MultiTrajectoryStateMerger_H
#define MultiTrajectoryStateMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class TrajectoryStateOnSurface;

/** Merging of MultiTrajectoryStates - uses MultiGaussianStateMergers
 *  for the actual merging.
 */

class MultiTrajectoryStateMerger {
public:
  MultiTrajectoryStateMerger(const MultiGaussianStateMerger<5>& merger) : theMultiStateMerger(merger.clone()) {}
  TrajectoryStateOnSurface merge(const TrajectoryStateOnSurface& tsos) const;
  MultiTrajectoryStateMerger* clone() const { return new MultiTrajectoryStateMerger(*this); }

private:
  const std::shared_ptr<const MultiGaussianStateMerger<5> > theMultiStateMerger;
};

#endif
