#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByChi2ESProducer.hh"
#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByHitsESProducer.hh"
#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByPositionESProducer.hh"
#include "SimTracker/TrackAssociatorESProducer/src/QuickTrackAssociatorByHitsESProducer.hh"

DEFINE_FWK_EVENTSETUP_MODULE(TrackAssociatorByHitsESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(TrackAssociatorByChi2ESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(TrackAssociatorByPositionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(QuickTrackAssociatorByHitsESProducer);
