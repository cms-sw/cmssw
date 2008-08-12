#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVAComputer.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVATrainer.h"

// define mva modules
DEFINE_FWK_MODULE(TtSemiJetCombMVAComputer);
DEFINE_FWK_MODULE(TtSemiJetCombMVATrainer);

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypGeom.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypWMassMaxSumPt.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypMaxSumPtWMass.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypGenMatch.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypMVADisc.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypKinFit.h"

// define event hypotheses
DEFINE_FWK_MODULE(TtSemiLepHypGeom);
DEFINE_FWK_MODULE(TtSemiLepHypWMassMaxSumPt);
DEFINE_FWK_MODULE(TtSemiLepHypMaxSumPtWMass);
DEFINE_FWK_MODULE(TtSemiLepHypGenMatch);
DEFINE_FWK_MODULE(TtSemiLepHypMVADisc);
DEFINE_FWK_MODULE(TtSemiLepHypKinFit);
