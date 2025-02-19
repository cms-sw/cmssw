#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"
#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"

#include "FWCore/Utilities/interface/typelookup.h"


TYPELOOKUP_DATA_REG(MultiGaussianStateMerger<5>);
TYPELOOKUP_DATA_REG(CloseComponentsMerger<5>);

TYPELOOKUP_DATA_REG(DistanceBetweenComponents<5>);
TYPELOOKUP_DATA_REG(KullbackLeiblerDistance<5>);

