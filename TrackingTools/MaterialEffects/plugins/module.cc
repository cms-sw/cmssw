#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "PropagatorWithMaterialESProducer.h"


#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(PropagatorWithMaterialESProducer);
