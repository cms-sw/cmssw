#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"


EVENTSETUP_DATA_REG(MultiGaussianStateMerger<5>);
EVENTSETUP_DATA_REG(CloseComponentsMerger<5>);

EVENTSETUP_DATA_REG(DistanceBetweenComponents<5>);
EVENTSETUP_DATA_REG(KullbackLeiblerDistance<5>);

