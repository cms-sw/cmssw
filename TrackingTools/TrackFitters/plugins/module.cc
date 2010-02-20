#include "TrackingTools/TrackFitters/plugins/KFTrajectoryFitterESProducer.h" 
#include "TrackingTools/TrackFitters/plugins/KFTrajectorySmootherESProducer.h" 
#include "TrackingTools/TrackFitters/plugins/KFFittingSmootherESProducer.h" 

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectoryFitterESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectorySmootherESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(KFFittingSmootherESProducer);
