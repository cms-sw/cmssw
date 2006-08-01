#include "TrackingTools/GsfTracking/interface/LargestWeightsTSOSMerger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "TrackingTools/GsfTools/src/TrajectoryStateLessWeight.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<algorithm>

LargestWeightsTSOSMerger::LargestWeightsTSOSMerger (int maxNrOfComponents,
						    bool smallestWeightsMerging) :
  theMaxNumberOfComponents(maxNrOfComponents),
  theSmallestWeightsMerging(smallestWeightsMerging) {}

TrajectoryStateOnSurface 
LargestWeightsTSOSMerger::merge(const TrajectoryStateOnSurface& tsos) const {
  
  //TrajectoryStateOnSurface trimmedTsos = KeepingNonZeroWeightsMerger().merge(tsos);
  
  std::vector<TrajectoryStateOnSurface> unmergedComponents = tsos.components();
  
  if (unmergedComponents.empty()) {
    edm::LogError("LargestWeightsTSOSMerger") << "Trying to merge trajectory state with zero components!";
    return TrajectoryStateOnSurface();
  }

  if (theMaxNumberOfComponents <= 0) {
    edm::LogError("LargestWeightsTSOSMerger") 
      << "Trying to merge state into zero (or less!) components, returning invalid state!";
    return TrajectoryStateOnSurface();
  }

  if (tsos.weight() == 0) {
    edm::LogError("LargestWeightsTSOSMerger") 
      << "Trying to merge mixture with sum of weights equal to zero!";
    return TrajectoryStateOnSurface();
  }

  if (unmergedComponents.size() < theMaxNumberOfComponents + 1)
    return TrajectoryStateOnSurface(new BasicMultiTrajectoryState(unmergedComponents));
  
  std::vector<double> weights;
  MultiTrajectoryStateAssembler result;
  std::vector<TrajectoryStateOnSurface> collapsableComponents;

  sort(unmergedComponents.begin(), unmergedComponents.end(), TrajectoryStateLessWeight());

  unsigned int nComp = 0;
  for (std::vector<TrajectoryStateOnSurface>::const_iterator iter = unmergedComponents.begin();
       iter != unmergedComponents.end(); iter++) {
    nComp++;
    if (nComp < theMaxNumberOfComponents) {
      result.addState(*iter);
    }
    else {
      collapsableComponents.push_back(*iter);
      if (!theSmallestWeightsMerging) break;
    }
  }

  result.addState(MultiTrajectoryStateCombiner().combine(collapsableComponents));

  return result.combinedState(1.);
}
