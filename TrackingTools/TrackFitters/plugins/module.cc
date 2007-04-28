#include "TrackingTools/TrackFitters/plugins/KFTrajectoryFitterESProducer.h" 
#include "TrackingTools/TrackFitters/plugins/KFTrajectorySmootherESProducer.h" 
#include "TrackingTools/TrackFitters/plugins/KFFittingSmootherESProducer.h" 

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectoryFitterESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(KFTrajectorySmootherESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(KFFittingSmootherESProducer);
