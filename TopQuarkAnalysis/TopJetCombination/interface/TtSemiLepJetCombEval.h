#ifndef TtSemiLepJetCombEval_h
#define TtSemiLepJetCombEval_h

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetComb.h"

// ----------------------------------------------------------------------
// common interface for the evaluation of multivariate analysis variables
// for jet combinations in semi leptonic ttbar decays
// used by the MVATrainer and MVAComputer
// ----------------------------------------------------------------------

inline double evaluateTtSemiLepJetComb(PhysicsTools::MVAComputerCache& mvaComputer,
				       const TtSemiLepJetComb& jetComb, const bool training=false, const bool trueCombi=false)
{
  // namespace for enums
  using namespace JetComb;
  std::vector<PhysicsTools::Variable::Value> values;
  
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kTargetId, trueCombi) );

  // ----------------------------------------------------------------------
  // mass, pt, eta, phi and theta of single candidates of the ttbar system
  // ----------------------------------------------------------------------
  // hadronic top quark
  values.push_back( PhysicsTools::Variable::Value("massHadTop" , jetComb.topVar(kHad, kMass ) ));
  values.push_back( PhysicsTools::Variable::Value("ptHadTop"   , jetComb.topVar(kHad, kPt   ) ));
  values.push_back( PhysicsTools::Variable::Value("etaHadTop"  , jetComb.topVar(kHad, kEta  ) ));
  values.push_back( PhysicsTools::Variable::Value("phiHadTop"  , jetComb.topVar(kHad, kPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("thetaHadTop", jetComb.topVar(kHad, kTheta) ));
  // leptonic top quark
  values.push_back( PhysicsTools::Variable::Value("massLepTop" , jetComb.topVar(kLep, kMass ) ));
  values.push_back( PhysicsTools::Variable::Value("ptLepTop"   , jetComb.topVar(kLep, kPt   ) ));
  values.push_back( PhysicsTools::Variable::Value("etaLepTop"  , jetComb.topVar(kLep, kEta  ) ));
  values.push_back( PhysicsTools::Variable::Value("phiLepTop"  , jetComb.topVar(kLep, kPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("thetaLepTop", jetComb.topVar(kLep, kTheta) ));
  // hadronic W boson
  values.push_back( PhysicsTools::Variable::Value("massHadW" , jetComb.wBosonVar(kHad, kMass ) ));
  values.push_back( PhysicsTools::Variable::Value("ptHadW"   , jetComb.wBosonVar(kHad, kPt   ) ));
  values.push_back( PhysicsTools::Variable::Value("etaHadW"  , jetComb.wBosonVar(kHad, kEta  ) ));
  values.push_back( PhysicsTools::Variable::Value("phiHadW"  , jetComb.wBosonVar(kHad, kPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("thetaHadW", jetComb.wBosonVar(kHad, kTheta) ));
  // hadronic b quark
  values.push_back( PhysicsTools::Variable::Value("ptHadB"   , jetComb.bQuarkVar(kHad, kPt   ) ));
  values.push_back( PhysicsTools::Variable::Value("etaHadB"  , jetComb.bQuarkVar(kHad, kEta  ) ));
  values.push_back( PhysicsTools::Variable::Value("phiHadB"  , jetComb.bQuarkVar(kHad, kPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("thetaHadB", jetComb.bQuarkVar(kHad, kTheta) ));
  // leptonic b quark
  values.push_back( PhysicsTools::Variable::Value("ptLepB"   , jetComb.bQuarkVar(kLep, kPt   ) ));
  values.push_back( PhysicsTools::Variable::Value("etaLepB"  , jetComb.bQuarkVar(kLep, kEta  ) ));
  values.push_back( PhysicsTools::Variable::Value("phiLepB"  , jetComb.bQuarkVar(kLep, kPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("thetaLepB", jetComb.bQuarkVar(kLep, kTheta) ));
  // light quark
  values.push_back( PhysicsTools::Variable::Value("ptLightQ"   , jetComb.lightQVar(false, kPt   ) ));
  values.push_back( PhysicsTools::Variable::Value("etaLightQ"  , jetComb.lightQVar(false, kEta  ) ));
  values.push_back( PhysicsTools::Variable::Value("phiLightQ"  , jetComb.lightQVar(false, kPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("thetaLightQ", jetComb.lightQVar(false, kTheta) ));
  // light anti-quark
  values.push_back( PhysicsTools::Variable::Value("ptLightQBar"   , jetComb.lightQVar(true, kPt   ) ));
  values.push_back( PhysicsTools::Variable::Value("etaLightQBar"  , jetComb.lightQVar(true, kEta  ) ));
  values.push_back( PhysicsTools::Variable::Value("phiLightQBar"  , jetComb.lightQVar(true, kPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("thetaLightQBar", jetComb.lightQVar(true, kTheta) ));
  // ----------------------------------------------------------------------
  // compare two candidates of the ttbar system in DeltaM, DeltaR, DeltaPhi or DeltaTheta
  // ----------------------------------------------------------------------
  // the two top quarks
  values.push_back( PhysicsTools::Variable::Value("deltaMHadTopLepTop"    , jetComb.compareHadTopLepTop(kDeltaM    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaRHadTopLepTop"    , jetComb.compareHadTopLepTop(kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadTopLepTop"  , jetComb.compareHadTopLepTop(kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopLepTop", jetComb.compareHadTopLepTop(kDeltaTheta) ));
  // the two W bosons
  values.push_back( PhysicsTools::Variable::Value("deltaMHadWLepW"    , jetComb.compareHadWLepW(kDeltaM    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaRHadWLepW"    , jetComb.compareHadWLepW(kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadWLepW"  , jetComb.compareHadWLepW(kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadWLepW", jetComb.compareHadWLepW(kDeltaTheta) ));
  // the two b quarks
  values.push_back( PhysicsTools::Variable::Value("deltaRHadBLepB"    , jetComb.compareHadBLepB(kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadBLepB"  , jetComb.compareHadBLepB(kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadBLepB", jetComb.compareHadBLepB(kDeltaTheta) ));
  // the two light quarks
  values.push_back( PhysicsTools::Variable::Value("deltaRHadQHadQBar"    , jetComb.compareLightQuarks(kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadQHadQBar"  , jetComb.compareLightQuarks(kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadQHadQBar", jetComb.compareLightQuarks(kDeltaTheta) ));
  // the hadronic top and the hadronic W
  values.push_back( PhysicsTools::Variable::Value("deltaMHadTopHadW"    , jetComb.compareTopW(kHad, kHad, kDeltaM    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaRHadTopHadW"    , jetComb.compareTopW(kHad, kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadTopHadW"  , jetComb.compareTopW(kHad, kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopHadW", jetComb.compareTopW(kHad, kHad, kDeltaTheta) ));
  // the leptonic top and the leptonic W
  values.push_back( PhysicsTools::Variable::Value("deltaMLepTopLepW"    , jetComb.compareTopW(kLep, kLep, kDeltaM    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaRLepTopLepW"    , jetComb.compareTopW(kLep, kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepTopLepW"  , jetComb.compareTopW(kLep, kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepTopLepW", jetComb.compareTopW(kLep, kLep, kDeltaTheta) ));
  // the hadronic top and the leptonic W
  values.push_back( PhysicsTools::Variable::Value("deltaMHadTopLepW"    , jetComb.compareTopW(kHad, kLep, kDeltaM    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaRHadTopLepW"    , jetComb.compareTopW(kHad, kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadTopLepW"  , jetComb.compareTopW(kHad, kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopLepW", jetComb.compareTopW(kHad, kLep, kDeltaTheta) ));
  // the leptonic top and the hadronic W
  values.push_back( PhysicsTools::Variable::Value("deltaMLepTopHadW"    , jetComb.compareTopW(kLep, kHad, kDeltaM    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaRLepTopHadW"    , jetComb.compareTopW(kLep, kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepTopHadW"  , jetComb.compareTopW(kLep, kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepTopHadW", jetComb.compareTopW(kLep, kHad, kDeltaTheta) ));
  // the hadronic top and the hadronic b
  values.push_back( PhysicsTools::Variable::Value("deltaRHadTopHadB"    , jetComb.compareTopB(kHad, kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadTopHadB"  , jetComb.compareTopB(kHad, kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopHadB", jetComb.compareTopB(kHad, kHad, kDeltaTheta) ));
  // the leptonic top and the leptonic b
  values.push_back( PhysicsTools::Variable::Value("deltaRLepTopLepB"    , jetComb.compareTopB(kLep, kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepTopLepB"  , jetComb.compareTopB(kLep, kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepTopLepB", jetComb.compareTopB(kLep, kLep, kDeltaTheta) ));
  // the hadronic top and the leptonic b
  values.push_back( PhysicsTools::Variable::Value("deltaRHadTopLepB"    , jetComb.compareTopB(kHad, kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadTopLepB"  , jetComb.compareTopB(kHad, kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopLepB", jetComb.compareTopB(kHad, kLep, kDeltaTheta) ));
  // the leptonic top and the hadronic b
  values.push_back( PhysicsTools::Variable::Value("deltaRLepTopHadB"    , jetComb.compareTopB(kLep, kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepTopHadB"  , jetComb.compareTopB(kLep, kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepTopHadB", jetComb.compareTopB(kLep, kHad, kDeltaTheta) ));
  // the hadronic W and the hadronic b
  values.push_back( PhysicsTools::Variable::Value("deltaRHadWHadB"    , jetComb.compareWB(kHad, kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadWHadB"  , jetComb.compareWB(kHad, kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadWHadB", jetComb.compareWB(kHad, kHad, kDeltaTheta) ));
  // the leptonic W and the leptonic b
  values.push_back( PhysicsTools::Variable::Value("deltaRLepWLepB"    , jetComb.compareWB(kLep, kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepWLepB"  , jetComb.compareWB(kLep, kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepWLepB", jetComb.compareWB(kLep, kLep, kDeltaTheta) ));
  // the hadronic W and the leptonic b
  values.push_back( PhysicsTools::Variable::Value("deltaRHadWLepB"    , jetComb.compareWB(kHad, kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadWLepB"  , jetComb.compareWB(kHad, kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadWLepB", jetComb.compareWB(kHad, kLep, kDeltaTheta) ));
  // the leptonic W and the hadronic b
  values.push_back( PhysicsTools::Variable::Value("deltaRLepWHadB"    , jetComb.compareWB(kLep, kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepWHadB"  , jetComb.compareWB(kLep, kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepWHadB", jetComb.compareWB(kLep, kHad, kDeltaTheta) ));
  // the hadronic top and the lepton
  values.push_back( PhysicsTools::Variable::Value("deltaRHadTopLepton"    , jetComb.compareTopLepton(kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadTopLepton"  , jetComb.compareTopLepton(kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopLepton", jetComb.compareTopLepton(kHad, kDeltaTheta) ));
  // the leptonic top and the lepton
  values.push_back( PhysicsTools::Variable::Value("deltaRLepTopLepton"    , jetComb.compareTopLepton(kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepTopLepton"  , jetComb.compareTopLepton(kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepTopLepton", jetComb.compareTopLepton(kLep, kDeltaTheta) ));
  // the hadronic top and the neutrino
  values.push_back( PhysicsTools::Variable::Value("deltaRHadTopNeutrino"    , jetComb.compareTopNeutrino(kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadTopNeutrino"  , jetComb.compareTopNeutrino(kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopNeutrino", jetComb.compareTopNeutrino(kHad, kDeltaTheta) ));
  // the leptonic top and the neutrino
  values.push_back( PhysicsTools::Variable::Value("deltaRLepTopNeutrino"    , jetComb.compareTopNeutrino(kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepTopNeutrino"  , jetComb.compareTopNeutrino(kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepTopNeutrino", jetComb.compareTopNeutrino(kLep, kDeltaTheta) ));
  // the hadronic W and the lepton
  values.push_back( PhysicsTools::Variable::Value("deltaRHadWLepton"    , jetComb.compareWLepton(kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadWLepton"  , jetComb.compareWLepton(kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadWLepton", jetComb.compareWLepton(kHad, kDeltaTheta) ));
  // the hadronic W and the neutrino
  values.push_back( PhysicsTools::Variable::Value("deltaRHadWNeutrino"    , jetComb.compareWNeutrino(kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadWNeutrino"  , jetComb.compareWNeutrino(kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadWNeutrino", jetComb.compareWNeutrino(kHad, kDeltaTheta) ));
  // the hadronic b and the lepton
  values.push_back( PhysicsTools::Variable::Value("deltaRHadBLepton"    , jetComb.compareBLepton(kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadBLepton"  , jetComb.compareBLepton(kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadBLepton", jetComb.compareBLepton(kHad, kDeltaTheta) ));
  // the leptonic b and the lepton
  values.push_back( PhysicsTools::Variable::Value("deltaRLepBLepton"    , jetComb.compareBLepton(kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepBLepton"  , jetComb.compareBLepton(kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepBLepton", jetComb.compareBLepton(kLep, kDeltaTheta) ));
  // the hadronic b and the neutrino
  values.push_back( PhysicsTools::Variable::Value("deltaRHadBNeutrino"    , jetComb.compareBNeutrino(kHad, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiHadBNeutrino"  , jetComb.compareBNeutrino(kHad, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadBNeutrino", jetComb.compareBNeutrino(kHad, kDeltaTheta) ));
  // the leptonic b and the neutrino
  values.push_back( PhysicsTools::Variable::Value("deltaRLepBNeutrino"    , jetComb.compareBNeutrino(kLep, kDeltaR    ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaPhiLepBNeutrino"  , jetComb.compareBNeutrino(kLep, kDeltaPhi  ) ));
  values.push_back( PhysicsTools::Variable::Value("deltaThetaLepBNeutrino", jetComb.compareBNeutrino(kLep, kDeltaTheta) ));
  // ----------------------------------------------------------------------
  // special variables combining the pt of the jets
  // ----------------------------------------------------------------------
  values.push_back( PhysicsTools::Variable::Value("relativePtHadronicTop", jetComb.relativePtHadronicTop() ));
  values.push_back( PhysicsTools::Variable::Value("bOverLightQPt"        , jetComb.bOverLightQPt()         ));
  // ----------------------------------------------------------------------
  // variables based on b-tagging with six different algorithms
  // ----------------------------------------------------------------------
  // hadronic b quark candidate
  values.push_back( PhysicsTools::Variable::Value("bTagHadBTrkCntHighEff", jetComb.bTag(kHad, kTrkCntHighEff) ));
  values.push_back( PhysicsTools::Variable::Value("bTagHadBTrkCntHighPur", jetComb.bTag(kHad, kTrkCntHighPur) ));
  values.push_back( PhysicsTools::Variable::Value("bTagHadBSoftMuon"     , jetComb.bTag(kHad, kSoftMuon     ) ));
  values.push_back( PhysicsTools::Variable::Value("bTagHadBSimpSecondVtx", jetComb.bTag(kHad, kSimpSecondVtx) ));
  values.push_back( PhysicsTools::Variable::Value("bTagHadBCombSecondVtx", jetComb.bTag(kHad, kCombSecondVtx) ));
  values.push_back( PhysicsTools::Variable::Value("bTagHadBImpactParaMVA", jetComb.bTag(kHad, kImpactParaMVA) ));
  // leptonic b quark candidate
  values.push_back( PhysicsTools::Variable::Value("bTagLepBTrkCntHighEff", jetComb.bTag(kLep, kTrkCntHighEff) ));
  values.push_back( PhysicsTools::Variable::Value("bTagLepBTrkCntHighPur", jetComb.bTag(kLep, kTrkCntHighPur) ));
  values.push_back( PhysicsTools::Variable::Value("bTagLepBSoftMuon"     , jetComb.bTag(kLep, kSoftMuon     ) ));
  values.push_back( PhysicsTools::Variable::Value("bTagLepBSimpSecondVtx", jetComb.bTag(kLep, kSimpSecondVtx) ));
  values.push_back( PhysicsTools::Variable::Value("bTagLepBCombSecondVtx", jetComb.bTag(kLep, kCombSecondVtx) ));
  values.push_back( PhysicsTools::Variable::Value("bTagLepBImpactParaMVA", jetComb.bTag(kLep, kImpactParaMVA) ));
  // summed b-tags of the two b quark candidates
  values.push_back( PhysicsTools::Variable::Value("bTagSumTrkCntHighEff", jetComb.combinedBTags(kTrkCntHighEff, kAdd) ));
  values.push_back( PhysicsTools::Variable::Value("bTagSumTrkCntHighPur", jetComb.combinedBTags(kTrkCntHighPur, kAdd) ));
  values.push_back( PhysicsTools::Variable::Value("bTagSumSoftMuon"     , jetComb.combinedBTags(kSoftMuon     , kAdd) ));
  values.push_back( PhysicsTools::Variable::Value("bTagSumSimpSecondVtx", jetComb.combinedBTags(kSimpSecondVtx, kAdd) ));
  values.push_back( PhysicsTools::Variable::Value("bTagSumCombSecondVtx", jetComb.combinedBTags(kCombSecondVtx, kAdd) ));
  values.push_back( PhysicsTools::Variable::Value("bTagSumImpactParaMVA", jetComb.combinedBTags(kImpactParaMVA, kAdd) ));
  // multiplied b-tags of the two b quark candidates
  values.push_back( PhysicsTools::Variable::Value("bTagProdImpactParaMVA", jetComb.combinedBTags(kImpactParaMVA, kMult) ));

  return mvaComputer->eval( values );
}

#endif
