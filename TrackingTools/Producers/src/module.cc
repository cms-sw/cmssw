#include "TrackingTools/Producers/interface/AnalyticalPropagatorESProducer.h"
#include "TrackingTools/Producers/interface/StraightLinePropagatorESProducer.h"
#include "TrackingTools/Producers/interface/SmartPropagatorESProducer.h"
#include "TrackingTools/Producers/interface/BeamHaloPropagatorESProducer.h"
#include "TrackingTools/Producers/interface/TrajectoryCleanerESProducer.h"
#include "TrackingTools/Producers/interface/TrajectoryFilterESProducer.h"
#include "TrackingTools/Producers/interface/CompositeTrajectoryFilterESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(StraightLinePropagatorESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(AnalyticalPropagatorESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SmartPropagatorESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(BeamHaloPropagatorESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(TrajectoryCleanerESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(TrajectoryFilterESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CompositeTrajectoryFilterESProducer);
