#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitterESProducer.h" 
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmootherESProducer.h" 
#include "TrackingTools/TrackFitters/interface/KFFittingSmootherESProducer.h" 

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectoryFitterESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(KFTrajectorySmootherESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(KFFittingSmootherESProducer);
