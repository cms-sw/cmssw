#include "TrackingTools/GsfTracking/plugins/GsfTrajectoryFitterESProducer.h"
#include "TrackingTools/GsfTracking/plugins/GsfTrajectorySmootherESProducer.h"
#include "TrackingTools/GsfTracking/plugins/GsfMaterialEffectsESProducer.h"
// #include "TrackingTools/GsfTracking/plugins/CloseComponentsTSOSMergerESProducer.h"
// #include "TrackingTools/GsfTracking/plugins/LargestWeightsTSOSMergerESProducer.h"
// #include "TrackingTools/GsfTracking/plugins/TSOSDistanceESProducer.h"

// #include "FWCore/Framework/interface/EventSetup.h"
// #include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
// #include "FWCore/Framework/interface/ESProducer.h"
// #include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(GsfTrajectoryFitterESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(GsfTrajectorySmootherESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(GsfMaterialEffectsESProducer);
// DEFINE_FWK_EVENTSETUP_MODULE(CloseComponentsTSOSMergerESProducer);
// DEFINE_FWK_EVENTSETUP_MODULE(LargestWeightsTSOSMergerESProducer);
// DEFINE_FWK_EVENTSETUP_MODULE(TSOSDistanceESProducer);

