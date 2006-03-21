#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitterESProducer.h" 
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(TrajectoryFitter);
DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectoryFitterESProducer)

