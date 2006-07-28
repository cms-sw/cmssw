#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilderESProducer.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h" 
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(TkTransientTrackingRecHitBuilderESProducer)
  // taken from TrackingTools EVENTSETUP_DATA_REG(TransientTrackingRecHitBuilder)
