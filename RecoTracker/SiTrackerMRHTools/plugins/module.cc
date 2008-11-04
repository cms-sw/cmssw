#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
//#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h" 
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "RecoTracker/SiTrackerMRHTools/plugins/SiTrackerMultiRecHitUpdatorESProducer.h"
#include "RecoTracker/SiTrackerMRHTools/plugins/MultiRecHitCollectorESProducer.h"
#include "RecoTracker/SiTrackerMRHTools/plugins/SiTrackerMultiRecHitUpdatorMTFESProducer.h"
#include "RecoTracker/SiTrackerMRHTools/plugins/MultiTrackFilterCollectorESProducer.h"

DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiTrackerMultiRecHitUpdatorESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MultiRecHitCollectorESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiTrackerMultiRecHitUpdatorMTFESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MultiTrackFilterCollectorESProducer);
