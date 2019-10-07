#include "FWCore/Framework/interface/MakerMacros.h"

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombMVAComputer.h"

// define mva modules
DEFINE_FWK_MODULE(TtSemiLepJetCombMVAComputer);

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombGeom.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombMaxSumPtWMass.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombWMassDeltaTopMass.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombWMassMaxSumPt.h"
DEFINE_FWK_MODULE(TtSemiLepJetCombGeom);
DEFINE_FWK_MODULE(TtSemiLepJetCombMaxSumPtWMass);
DEFINE_FWK_MODULE(TtSemiLepJetCombWMassDeltaTopMass);
DEFINE_FWK_MODULE(TtSemiLepJetCombWMassMaxSumPt);

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypGeom.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypWMassDeltaTopMass.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypWMassMaxSumPt.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypMaxSumPtWMass.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypGenMatch.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypMVADisc.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypKinFit.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypHitFit.h"

// define event hypotheses
DEFINE_FWK_MODULE(TtSemiLepHypGeom);
DEFINE_FWK_MODULE(TtSemiLepHypWMassDeltaTopMass);
DEFINE_FWK_MODULE(TtSemiLepHypWMassMaxSumPt);
DEFINE_FWK_MODULE(TtSemiLepHypMaxSumPtWMass);
DEFINE_FWK_MODULE(TtSemiLepHypGenMatch);
DEFINE_FWK_MODULE(TtSemiLepHypMVADisc);
DEFINE_FWK_MODULE(TtSemiLepHypKinFit);
DEFINE_FWK_MODULE(TtSemiLepHypHitFit);

// define fully leptonic event hypotheses
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullLepHypGenMatch.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullLepHypKinSolution.h"

DEFINE_FWK_MODULE(TtFullLepHypGenMatch);
DEFINE_FWK_MODULE(TtFullLepHypKinSolution);

// define fully hadronic event hypotheses
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullHadHypGenMatch.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullHadHypKinFit.h"
DEFINE_FWK_MODULE(TtFullHadHypGenMatch);
DEFINE_FWK_MODULE(TtFullHadHypKinFit);
