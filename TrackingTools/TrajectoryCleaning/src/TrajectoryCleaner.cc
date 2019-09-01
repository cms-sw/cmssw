
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include <cassert>

void TrajectoryCleaner::clean(TempTrajectoryContainer&) const {
  edm::LogError("TrajectoryCleaner") << "not implemented for TempTrajectory";
  assert(false);
}

void TrajectoryCleaner::clean(TrajectoryContainer& tc) const {
  TrajectoryPointerContainer thePointerContainer;
  thePointerContainer.reserve(tc.size());
  for (TrajectoryCleaner::TrajectoryIterator it = tc.begin(); it != tc.end(); it++) {
    thePointerContainer.push_back(&(*it));
  }

  clean(thePointerContainer);
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(TrajectoryCleaner);
