//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchSeedFinderPlugin
// 
// Description:     plugin for RoadSearchSeedFinder
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed May 23 19:20:34 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/03/07 21:46:50 $
// $Revision: 1.4 $
//

#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinder.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadSearchSeedFinder);
