#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoTau/interface/TauTagVal.h"
#include "Validation/RecoTau/interface/TauTagVal_BKG.h"
#include "Validation/RecoTau/interface/TauTagVal_EMIso.h"
#include "Validation/RecoTau/interface/TauTagVal_EMIso_BKG.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( TauTagVal );
DEFINE_ANOTHER_FWK_MODULE( TauTagVal_BKG );
DEFINE_ANOTHER_FWK_MODULE( TauTagVal_EMIso );
DEFINE_ANOTHER_FWK_MODULE( TauTagVal_EMIso_BKG );

