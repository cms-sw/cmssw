#include "TrackingTools/GsfTools/interface/LargestWeightsStateMerger.h"

//#include "TrackerReco/GsfPattern/interface/KeepingNonZeroWeightsMerger.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateAssembler.h"
#include "TrackingTools/GsfTools/src/GaussianStateLessWeight.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<algorithm>

// void
// LargestWeightsStateMerger::initConfigurables() 
// {
//   // parameter (could be configurable)
//   static SimpleConfigurable<bool> 
//     mergeConf(false,"LargestWeightsStateMerger:mergeSmallestComponents");
//   theSmallestWeightsMerging = mergeConf.value();
// }


RCMultiGaussianState 
LargestWeightsStateMerger::merge(const RCMultiGaussianState& mgs) const
{
// ThS: Can not check for TSOS invalidity
//   if (!tsos.isValid()) {
//     return tsos;
//   } else {
    
    //TSOS trimmedTsos = KeepingNonZeroWeightsMerger().merge(tsos);

  SGSVector unmergedComponents = mgs->components();
  SGSVector finalComponents;
    
  if (unmergedComponents.empty()) {
    edm::LogError("LargestWeightsStateMerger") 
      << "Trying to merge trajectory state with zero components!";
    return mgs; // ThS: TSOS();
  }

// ThS: Don't you want to throw an exception at construction of the class?
  if (Nmax <= 0) {
    edm::LogError("LargestWeightsStateMerger") 
      << "Trying to merge state into zero (or less!) components, returning invalid state!";
    return mgs; // ThS: TSOS();
  }
    
// ThS: Of course, here the TSOS will not be invalid. But it will have 0 components
  if (mgs->weight() == 0) {
    edm::LogInfo("LargestWeightsStateMerger") 
      << "Trying to merge mixture with sum of weights equal to zero!";
    return mgs->createNewState(finalComponents);
  }

  //TrajectoryStateOnSurface trimmedTsos = KeepingNonZeroWeightsMerger().merge(tsos);


  if ((int) unmergedComponents.size() < Nmax + 1) return mgs;
// ThS: Why not the initial object, as above?
//    return TrajectoryStateOnSurface(new BasicMultiTrajectoryState(unmergedComponents));
  
  std::vector<double> weights;
  MultiGaussianStateAssembler result(mgs);
  SGSVector collapsableComponents;

  sort(unmergedComponents.begin(), unmergedComponents.end(), GaussianStateLessWeight());

  int nComp = 0;
  for (SGSVector::const_iterator iter = unmergedComponents.begin();
       iter != unmergedComponents.end(); iter++) {
    nComp++;
    if (nComp < Nmax) {
      result.addState(*iter);
    }
    else {
      collapsableComponents.push_back(*iter);
      if (!theSmallestWeightsMerging) break;
    }
  }

  result.addState(MultiGaussianStateCombiner().combine(collapsableComponents));

  return result.combinedState();
}
