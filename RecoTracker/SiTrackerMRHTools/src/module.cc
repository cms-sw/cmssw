
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdatorMTF.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrackFilterHitCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

TYPELOOKUP_DATA_REG(SiTrackerMultiRecHitUpdator);
TYPELOOKUP_DATA_REG(SiTrackerMultiRecHitUpdatorMTF);
TYPELOOKUP_DATA_REG(MultiRecHitCollector);
TYPELOOKUP_DATA_REG(MultiTrackFilterHitCollector);
