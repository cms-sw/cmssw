#include "TrackingTools/GsfTools/interface/KeepingNonZeroWeightsMerger.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

RCMultiGaussianState
KeepingNonZeroWeightsMerger::merge(const RCMultiGaussianState& mgs) const {

  SGSVector unmergedComponents = mgs->components();

  if (unmergedComponents.empty()) {
    edm::LogError("KeepingNonZeroWeightsMerger") 
      << "Trying to merge trajectory state with zero components!";
    return mgs; // ThS: TSOS();
  }

// ThS: Don't you want to throw an exception at construction of the class?

  if (cut <= 0) {
    edm::LogError("KeepingNonZeroWeightsMerger")  
      << "Trying to merge state with cut value <= 0; returning invalid state!";
    return mgs; // ThS: TrajectoryStateOnSurface();
  }

  SGSVector finalComponents;
  double sumWeights = 0.;

  for (SGSVector::const_iterator iter = unmergedComponents.begin();
       iter != unmergedComponents.end(); iter++) {
    double weight = (**iter).weight();
    if (weight > cut) {
      sumWeights += weight;
    }
  }

// ThS: Of course, here the TSOS will not be invalid. But it will have 0 components
  if (sumWeights == 0) {
    edm::LogInfo("KeepingNonZeroWeightsMerger")  
      << "Trying to merge mixture with sum of weights equal to zero!";
    return mgs->createNewState(finalComponents);
  }

  for (SGSVector::const_iterator iter = unmergedComponents.begin();
       iter != unmergedComponents.end(); iter++) {
    double weight = (**iter).weight();
    if (weight > cut) {
      finalComponents.push_back((**iter).createNewState((**iter).mean(),
      				  (**iter).covariance(), weight/sumWeights ));
    }
  }

  return mgs->createNewState(finalComponents);
}
