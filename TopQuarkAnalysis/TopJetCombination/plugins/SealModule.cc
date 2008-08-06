#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVAComputer.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVATrainer.h"

// define mva modules
DEFINE_FWK_MODULE(TtSemiJetCombMVAComputer);
DEFINE_FWK_MODULE(TtSemiJetCombMVATrainer);


#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisMaxSumPtWMass.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisKinFit.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisGenMatch.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisMVADisc.h"

// define event hypotheses
DEFINE_FWK_MODULE(TtSemiHypothesisMaxSumPtWMass);
DEFINE_FWK_MODULE(TtSemiHypothesisKinFit);
DEFINE_FWK_MODULE(TtSemiHypothesisGenMatch);
DEFINE_FWK_MODULE(TtSemiHypothesisMVADisc);
