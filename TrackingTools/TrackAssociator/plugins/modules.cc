#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DetIdAssociatorFactory.h"
#include "EcalDetIdAssociatorMaker.h"
#include "HcalDetIdAssociatorMaker.h"
#include "HODetIdAssociatorMaker.h"
#include "CaloDetIdAssociatorMaker.h"
#include "MuonDetIdAssociatorMaker.h"
#include "PreshowerDetIdAssociatorMaker.h"

DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, EcalDetIdAssociatorMaker, "EcalDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, HcalDetIdAssociatorMaker, "HcalDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, HODetIdAssociatorMaker, "HODetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, CaloDetIdAssociatorMaker, "CaloDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, MuonDetIdAssociatorMaker, "MuonDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, PreshowerDetIdAssociatorMaker, "PreshowerDetIdAssociator");
