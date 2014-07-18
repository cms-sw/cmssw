#include "TrackingTools/Producers/interface/AnalyticalPropagatorESProducer.h"
#include "TrackingTools/Producers/interface/StraightLinePropagatorESProducer.h"
#include "TrackingTools/Producers/interface/SmartPropagatorESProducer.h"
#include "TrackingTools/Producers/interface/BeamHaloPropagatorESProducer.h"
#include "TrackingTools/Producers/interface/TrajectoryCleanerESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(StraightLinePropagatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(AnalyticalPropagatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(SmartPropagatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(BeamHaloPropagatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(TrajectoryCleanerESProducer);
