#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerFactory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
DEFINE_EDM_PLUGIN(TrajectoryCleanerFactory, TrajectoryCleanerBySharedHits, "TrajectoryCleanerBySharedHits");
