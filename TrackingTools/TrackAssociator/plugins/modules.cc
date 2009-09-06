#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "TrackingTools/TrackAssociator/interface/DetIdAssociatorFactory.h"
#include "TrackingTools/TrackAssociator/interface/EcalDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/HcalDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/HODetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/MuonDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/PreshowerDetIdAssociator.h"

DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, EcalDetIdAssociator, "EcalDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, HcalDetIdAssociator, "HcalDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, HODetIdAssociator, "HODetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, CaloDetIdAssociator, "CaloDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, MuonDetIdAssociator, "MuonDetIdAssociator");
DEFINE_EDM_PLUGIN(DetIdAssociatorFactory, PreshowerDetIdAssociator, "PreshowerDetIdAssociator");
