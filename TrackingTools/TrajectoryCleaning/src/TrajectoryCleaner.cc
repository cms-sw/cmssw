
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"


void TrajectoryCleaner::clean( TrajectoryContainer& tc) const
{
  TrajectoryPointerContainer thePointerContainer;
  for (TrajectoryCleaner::TrajectoryIterator it = tc.begin(); it != tc.end(); it++) {
    thePointerContainer.push_back( &(*it) );
  }

  clean(thePointerContainer);
}
