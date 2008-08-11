#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVAComputer.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVATrainer.h"

// define mva modules
DEFINE_FWK_MODULE(TtSemiJetCombMVAComputer);
DEFINE_FWK_MODULE(TtSemiJetCombMVATrainer);


#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepMaxSumPtWMass.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepGenMatch.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepMVADisc.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepKinFit.h"


// define event hypotheses
DEFINE_FWK_MODULE(TtSemiLepMaxSumPtWMass);
DEFINE_FWK_MODULE(TtSemiLepGenMatch);
DEFINE_FWK_MODULE(TtSemiLepMVADisc);
DEFINE_FWK_MODULE(TtSemiLepKinFit);
