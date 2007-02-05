#include "TrackingTools/GsfTools/interface/CloseComponentsMergerESProducer.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponentsESProducer.h"

#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(CloseComponentsMerger);
EVENTSETUP_DATA_REG(DistanceBetweenComponents);

DEFINE_FWK_EVENTSETUP_MODULE(CloseComponentsMergerESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(DistanceBetweenComponentsESProducer);

