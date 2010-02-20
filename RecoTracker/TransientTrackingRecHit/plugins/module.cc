#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/plugins/TkTransientTrackingRecHitBuilderESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h" 
#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(TkTransientTrackingRecHitBuilderESProducer);

