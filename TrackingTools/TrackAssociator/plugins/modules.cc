#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(DetIdAssociator);


#include "DetIdAssociatorFactory.h"
#include "EcalDetIdAssociator.h"
#include "HcalDetIdAssociator.h"
#include "HODetIdAssociator.h"
#include "CaloDetIdAssociator.h"
#include "MuonDetIdAssociator.h"
#include "PreshowerDetIdAssociator.h"

DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, EcalDetIdAssociator, "EcalDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, HcalDetIdAssociator, "HcalDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, HODetIdAssociator, "HODetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, CaloDetIdAssociator, "CaloDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, MuonDetIdAssociator, "MuonDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, PreshowerDetIdAssociator, "PreshowerDetIdAssociator");

