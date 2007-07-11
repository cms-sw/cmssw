#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByChi2ESProducer.hh"
#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByHitsESProducer.hh"
#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByPositionESProducer.hh"

DEFINE_FWK_EVENTSETUP_MODULE(TrackAssociatorByHitsESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(TrackAssociatorByChi2ESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(TrackAssociatorByPositionESProducer);
