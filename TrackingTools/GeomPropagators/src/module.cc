#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagatorESProducer.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagatorESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(Propagator);
DEFINE_FWK_EVENTSETUP_MODULE(StraightLinePropagatorESProducer)
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(AnalyticalPropagatorESProducer)

