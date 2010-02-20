#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h" 
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h" 
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"


TYPELOOKUP_DATA_REG(TrajectoryBuilder);
TYPELOOKUP_DATA_REG(TrajectoryFitter);
TYPELOOKUP_DATA_REG(TrajectorySmoother);
TYPELOOKUP_DATA_REG(TrajectoryStateClosestToBeamLineBuilder);
