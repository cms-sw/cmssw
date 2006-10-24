#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitterESProducer.h" 
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmootherESProducer.h" 
#include "TrackingTools/TrackFitters/interface/KFFittingSmootherESProducer.h" 
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h" 
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h" 

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(TrajectoryFitter);
EVENTSETUP_DATA_REG(TrajectorySmoother);
DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectoryFitterESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(KFTrajectorySmootherESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(KFFittingSmootherESProducer);
