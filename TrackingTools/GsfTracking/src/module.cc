#include "TrackingTools/GsfTracking/interface/GsfTrajectoryFitterESProducer.h"
#include "TrackingTools/GsfTracking/interface/GsfTrajectorySmootherESProducer.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsESProducer.h"
#include "TrackingTools/GsfTracking/interface/CloseComponentsTSOSMergerESProducer.h"
#include "TrackingTools/GsfTracking/interface/LargestWeightsTSOSMergerESProducer.h"
#include "TrackingTools/GsfTracking/interface/TSOSDistanceESProducer.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h" 
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h" 
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "TrackingTools/GsfTracking/interface/TSOSDistanceBetweenComponents.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

// EVENTSETUP_DATA_REG(TrajectoryFitter);
// EVENTSETUP_DATA_REG(TrajectorySmoother);
EVENTSETUP_DATA_REG(GsfMaterialEffectsUpdator);
EVENTSETUP_DATA_REG(MultiTrajectoryStateMerger);
EVENTSETUP_DATA_REG(TSOSDistanceBetweenComponents);

DEFINE_FWK_EVENTSETUP_MODULE(GsfTrajectoryFitterESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(GsfTrajectorySmootherESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(GsfMaterialEffectsESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CloseComponentsTSOSMergerESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(LargestWeightsTSOSMergerESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(TSOSDistanceESProducer);

