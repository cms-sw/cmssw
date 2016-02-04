#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"



#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerFactory.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedSeeds.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerMerger.h"

DEFINE_EDM_PLUGIN(TrajectoryCleanerFactory, TrajectoryCleanerBySharedHits, "TrajectoryCleanerBySharedHits");
DEFINE_EDM_PLUGIN(TrajectoryCleanerFactory, TrajectoryCleanerBySharedSeeds, "TrajectoryCleanerBySharedSeeds");
DEFINE_EDM_PLUGIN(TrajectoryCleanerFactory, TrajectoryCleanerMerger, "TrajectoryCleanerMerger");
