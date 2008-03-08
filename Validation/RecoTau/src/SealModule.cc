#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoTau/interface/TauTagVal.h"
#include "Validation/RecoTau/interface/PFTauTagVal.h"
#include "Validation/RecoTau/interface/CaloTauTagVal.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( TauTagVal );
DEFINE_ANOTHER_FWK_MODULE( PFTauTagVal );
DEFINE_ANOTHER_FWK_MODULE( CaloTauTagVal );
