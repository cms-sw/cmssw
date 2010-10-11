
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "FWCore/Utilities/interface/typelookup.h"

void TrajectoryCleaner::clean( TrajectoryContainer& tc) const
{
  TrajectoryPointerContainer thePointerContainer;
  thePointerContainer.reserve(tc.size());
  for (TrajectoryCleaner::TrajectoryIterator it = tc.begin(); it != tc.end(); it++) {
    thePointerContainer.push_back( &(*it) );
  }

  clean(thePointerContainer);
}

TYPELOOKUP_DATA_REG(TrajectoryCleaner);
