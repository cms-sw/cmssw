
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"


void TrajectoryCleaner::clean( TrajectoryContainer& tc) const
{
  TrajectoryPointerContainer thePointerContainer;
  thePointerContainer.reserve(tc.size());
  for (TrajectoryCleaner::TrajectoryIterator it = tc.begin(); it != tc.end(); it++) {
    thePointerContainer.push_back( &(*it) );
  }

  clean(thePointerContainer);
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(TrajectoryCleaner);
