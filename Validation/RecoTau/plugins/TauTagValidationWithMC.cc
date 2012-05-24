#define MC_ENABLER

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DQMOffline/Tau/interface/TauTagValidation.h"
#include "Validation/RecoTau/interface/MCTauValidation.h"

typedef TauTagValidation<MCTauValidation>  TauTagValidationWithMC;

DEFINE_FWK_MODULE( TauTagValidationWithMC );

