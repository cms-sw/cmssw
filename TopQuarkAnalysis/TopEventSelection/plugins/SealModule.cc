#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtFullHadSignalSelMVAComputer.h"
#include "TopQuarkAnalysis/TopEventSelection/plugins/TtFullHadSignalSelMVATrainer.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVAComputer.h"
#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVATrainer.h"

// define mva modules
DEFINE_FWK_MODULE(TtFullHadSignalSelMVAComputer);
DEFINE_FWK_MODULE(TtFullHadSignalSelMVATrainer);

DEFINE_FWK_MODULE(TtSemiLepSignalSelMVAComputer);
DEFINE_FWK_MODULE(TtSemiLepSignalSelMVATrainer);

