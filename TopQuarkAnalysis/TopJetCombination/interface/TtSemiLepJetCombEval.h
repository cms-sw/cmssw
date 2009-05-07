#ifndef TtSemiLepJetCombEval_h
#define TtSemiLepJetCombEval_h

#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetComb.h"

// ----------------------------------------------------------------------
// common interface for the evaluation of multivariate analysis variables
// for jet combinations in semi leptonic ttbar decays
// used by the MVATrainer and MVAComputer
// ----------------------------------------------------------------------

inline void evaluateTtSemiLepJetComb(PhysicsTools::Variable::ValueList& values, const TtSemiLepJetComb& jetComb)
{
  // namespace for enums
  using namespace JetComb;
  
  // ----------------------------------------------------------------------
  // mass, pt, eta, phi and theta of single candidates of the ttbar system
  // ----------------------------------------------------------------------
  // hadronic top quark
  values.add( "massHadTop" , jetComb.topVar(kHad, kMass ) );
  values.add( "ptHadTop"   , jetComb.topVar(kHad, kPt   ) );
  values.add( "etaHadTop"  , jetComb.topVar(kHad, kEta  ) );
  values.add( "phiHadTop"  , jetComb.topVar(kHad, kPhi  ) );
  values.add( "thetaHadTop", jetComb.topVar(kHad, kTheta) );
  // leptonic top quark
  values.add( "massLepTop" , jetComb.topVar(kLep, kMass ) );
  values.add( "ptLepTop"   , jetComb.topVar(kLep, kPt   ) );
  values.add( "etaLepTop"  , jetComb.topVar(kLep, kEta  ) );
  values.add( "phiLepTop"  , jetComb.topVar(kLep, kPhi  ) );
  values.add( "thetaLepTop", jetComb.topVar(kLep, kTheta) );
  // hadronic W boson
  values.add( "massHadW" , jetComb.wBosonVar(kHad, kMass ) );
  values.add( "ptHadW"   , jetComb.wBosonVar(kHad, kPt   ) );
  values.add( "etaHadW"  , jetComb.wBosonVar(kHad, kEta  ) );
  values.add( "phiHadW"  , jetComb.wBosonVar(kHad, kPhi  ) );
  values.add( "thetaHadW", jetComb.wBosonVar(kHad, kTheta) );
  // hadronic b quark
  values.add( "ptHadB"   , jetComb.bQuarkVar(kHad, kPt   ) );
  values.add( "etaHadB"  , jetComb.bQuarkVar(kHad, kEta  ) );
  values.add( "phiHadB"  , jetComb.bQuarkVar(kHad, kPhi  ) );
  values.add( "thetaHadB", jetComb.bQuarkVar(kHad, kTheta) );
  // leptonic b quark
  values.add( "ptLepB"   , jetComb.bQuarkVar(kLep, kPt   ) );
  values.add( "etaLepB"  , jetComb.bQuarkVar(kLep, kEta  ) );
  values.add( "phiLepB"  , jetComb.bQuarkVar(kLep, kPhi  ) );
  values.add( "thetaLepB", jetComb.bQuarkVar(kLep, kTheta) );
  // light quark
  values.add( "ptLightQ"   , jetComb.lightQVar(false, kPt   ) );
  values.add( "etaLightQ"  , jetComb.lightQVar(false, kEta  ) );
  values.add( "phiLightQ"  , jetComb.lightQVar(false, kPhi  ) );
  values.add( "thetaLightQ", jetComb.lightQVar(false, kTheta) );
  // light anti-quark
  values.add( "ptLightQBar"   , jetComb.lightQVar(true, kPt   ) );
  values.add( "etaLightQBar"  , jetComb.lightQVar(true, kEta  ) );
  values.add( "phiLightQBar"  , jetComb.lightQVar(true, kPhi  ) );
  values.add( "thetaLightQBar", jetComb.lightQVar(true, kTheta) );
  // ----------------------------------------------------------------------
  // compare two candidates of the ttbar system in DeltaM, DeltaR, DeltaPhi or DeltaTheta
  // ----------------------------------------------------------------------
  // the two top quarks
  values.add( "deltaMHadTopLepTop"    , jetComb.compareHadTopLepTop(kDeltaM    ) );
  values.add( "deltaRHadTopLepTop"    , jetComb.compareHadTopLepTop(kDeltaR    ) );
  values.add( "deltaPhiHadTopLepTop"  , jetComb.compareHadTopLepTop(kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepTop", jetComb.compareHadTopLepTop(kDeltaTheta) );
  // the two W bosons
  values.add( "deltaMHadWLepW"    , jetComb.compareHadWLepW(kDeltaM    ) );
  values.add( "deltaRHadWLepW"    , jetComb.compareHadWLepW(kDeltaR    ) );
  values.add( "deltaPhiHadWLepW"  , jetComb.compareHadWLepW(kDeltaPhi  ) );
  values.add( "deltaThetaHadWLepW", jetComb.compareHadWLepW(kDeltaTheta) );
  // the two b quarks
  values.add( "deltaRHadBLepB"    , jetComb.compareHadBLepB(kDeltaR    ) );
  values.add( "deltaPhiHadBLepB"  , jetComb.compareHadBLepB(kDeltaPhi  ) );
  values.add( "deltaThetaHadBLepB", jetComb.compareHadBLepB(kDeltaTheta) );
  // the two light quarks
  values.add( "deltaRHadQHadQBar"    , jetComb.compareLightQuarks(kDeltaR    ) );
  values.add( "deltaPhiHadQHadQBar"  , jetComb.compareLightQuarks(kDeltaPhi  ) );
  values.add( "deltaThetaHadQHadQBar", jetComb.compareLightQuarks(kDeltaTheta) );
  // the hadronic top and the hadronic W
  values.add( "deltaMHadTopHadW"    , jetComb.compareTopW(kHad, kHad, kDeltaM    ) );
  values.add( "deltaRHadTopHadW"    , jetComb.compareTopW(kHad, kHad, kDeltaR    ) );
  values.add( "deltaPhiHadTopHadW"  , jetComb.compareTopW(kHad, kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadTopHadW", jetComb.compareTopW(kHad, kHad, kDeltaTheta) );
  // the leptonic top and the leptonic W
  values.add( "deltaMLepTopLepW"    , jetComb.compareTopW(kLep, kLep, kDeltaM    ) );
  values.add( "deltaRLepTopLepW"    , jetComb.compareTopW(kLep, kLep, kDeltaR    ) );
  values.add( "deltaPhiLepTopLepW"  , jetComb.compareTopW(kLep, kLep, kDeltaPhi  ) );
  values.add( "deltaThetaLepTopLepW", jetComb.compareTopW(kLep, kLep, kDeltaTheta) );
  // the hadronic top and the leptonic W
  values.add( "deltaMHadTopLepW"    , jetComb.compareTopW(kHad, kLep, kDeltaM    ) );
  values.add( "deltaRHadTopLepW"    , jetComb.compareTopW(kHad, kLep, kDeltaR    ) );
  values.add( "deltaPhiHadTopLepW"  , jetComb.compareTopW(kHad, kLep, kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepW", jetComb.compareTopW(kHad, kLep, kDeltaTheta) );
  // the leptonic top and the hadronic W
  values.add( "deltaMLepTopHadW"    , jetComb.compareTopW(kLep, kHad, kDeltaM    ) );
  values.add( "deltaRLepTopHadW"    , jetComb.compareTopW(kLep, kHad, kDeltaR    ) );
  values.add( "deltaPhiLepTopHadW"  , jetComb.compareTopW(kLep, kHad, kDeltaPhi  ) );
  values.add( "deltaThetaLepTopHadW", jetComb.compareTopW(kLep, kHad, kDeltaTheta) );
  // the hadronic top and the hadronic b
  values.add( "deltaRHadTopHadB"    , jetComb.compareTopB(kHad, kHad, kDeltaR    ) );
  values.add( "deltaPhiHadTopHadB"  , jetComb.compareTopB(kHad, kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadTopHadB", jetComb.compareTopB(kHad, kHad, kDeltaTheta) );
  // the leptonic top and the leptonic b
  values.add( "deltaRLepTopLepB"    , jetComb.compareTopB(kLep, kLep, kDeltaR    ) );
  values.add( "deltaPhiLepTopLepB"  , jetComb.compareTopB(kLep, kLep, kDeltaPhi  ) );
  values.add( "deltaThetaLepTopLepB", jetComb.compareTopB(kLep, kLep, kDeltaTheta) );
  // the hadronic top and the leptonic b
  values.add( "deltaRHadTopLepB"    , jetComb.compareTopB(kHad, kLep, kDeltaR    ) );
  values.add( "deltaPhiHadTopLepB"  , jetComb.compareTopB(kHad, kLep, kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepB", jetComb.compareTopB(kHad, kLep, kDeltaTheta) );
  // the leptonic top and the hadronic b
  values.add( "deltaRLepTopHadB"    , jetComb.compareTopB(kLep, kHad, kDeltaR    ) );
  values.add( "deltaPhiLepTopHadB"  , jetComb.compareTopB(kLep, kHad, kDeltaPhi  ) );
  values.add( "deltaThetaLepTopHadB", jetComb.compareTopB(kLep, kHad, kDeltaTheta) );
  // the hadronic W and the hadronic b
  values.add( "deltaRHadWHadB"    , jetComb.compareWB(kHad, kHad, kDeltaR    ) );
  values.add( "deltaPhiHadWHadB"  , jetComb.compareWB(kHad, kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadWHadB", jetComb.compareWB(kHad, kHad, kDeltaTheta) );
  // the leptonic W and the leptonic b
  values.add( "deltaRLepWLepB"    , jetComb.compareWB(kLep, kLep, kDeltaR    ) );
  values.add( "deltaPhiLepWLepB"  , jetComb.compareWB(kLep, kLep, kDeltaPhi  ) );
  values.add( "deltaThetaLepWLepB", jetComb.compareWB(kLep, kLep, kDeltaTheta) );
  // the hadronic W and the leptonic b
  values.add( "deltaRHadWLepB"    , jetComb.compareWB(kHad, kLep, kDeltaR    ) );
  values.add( "deltaPhiHadWLepB"  , jetComb.compareWB(kHad, kLep, kDeltaPhi  ) );
  values.add( "deltaThetaHadWLepB", jetComb.compareWB(kHad, kLep, kDeltaTheta) );
  // the leptonic W and the hadronic b
  values.add( "deltaRLepWHadB"    , jetComb.compareWB(kLep, kHad, kDeltaR    ) );
  values.add( "deltaPhiLepWHadB"  , jetComb.compareWB(kLep, kHad, kDeltaPhi  ) );
  values.add( "deltaThetaLepWHadB", jetComb.compareWB(kLep, kHad, kDeltaTheta) );
  // the hadronic top and the lepton
  values.add( "deltaRHadTopLepton"    , jetComb.compareTopLepton(kHad, kDeltaR    ) );
  values.add( "deltaPhiHadTopLepton"  , jetComb.compareTopLepton(kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepton", jetComb.compareTopLepton(kHad, kDeltaTheta) );
  // the leptonic top and the lepton
  values.add( "deltaRLepTopLepton"    , jetComb.compareTopLepton(kLep, kDeltaR    ) );
  values.add( "deltaPhiLepTopLepton"  , jetComb.compareTopLepton(kLep, kDeltaPhi  ) );
  values.add( "deltaThetaLepTopLepton", jetComb.compareTopLepton(kLep, kDeltaTheta) );
  // the hadronic top and the neutrino
  values.add( "deltaRHadTopNeutrino"    , jetComb.compareTopNeutrino(kHad, kDeltaR    ) );
  values.add( "deltaPhiHadTopNeutrino"  , jetComb.compareTopNeutrino(kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadTopNeutrino", jetComb.compareTopNeutrino(kHad, kDeltaTheta) );
  // the leptonic top and the neutrino
  values.add( "deltaRLepTopNeutrino"    , jetComb.compareTopNeutrino(kLep, kDeltaR    ) );
  values.add( "deltaPhiLepTopNeutrino"  , jetComb.compareTopNeutrino(kLep, kDeltaPhi  ) );
  values.add( "deltaThetaLepTopNeutrino", jetComb.compareTopNeutrino(kLep, kDeltaTheta) );
  // the hadronic W and the lepton
  values.add( "deltaRHadWLepton"    , jetComb.compareWLepton(kHad, kDeltaR    ) );
  values.add( "deltaPhiHadWLepton"  , jetComb.compareWLepton(kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadWLepton", jetComb.compareWLepton(kHad, kDeltaTheta) );
  // the hadronic W and the neutrino
  values.add( "deltaRHadWNeutrino"    , jetComb.compareWNeutrino(kHad, kDeltaR    ) );
  values.add( "deltaPhiHadWNeutrino"  , jetComb.compareWNeutrino(kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadWNeutrino", jetComb.compareWNeutrino(kHad, kDeltaTheta) );
  // the hadronic b and the lepton
  values.add( "deltaRHadBLepton"    , jetComb.compareBLepton(kHad, kDeltaR    ) );
  values.add( "deltaPhiHadBLepton"  , jetComb.compareBLepton(kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadBLepton", jetComb.compareBLepton(kHad, kDeltaTheta) );
  // the leptonic b and the lepton
  values.add( "deltaRLepBLepton"    , jetComb.compareBLepton(kLep, kDeltaR    ) );
  values.add( "deltaPhiLepBLepton"  , jetComb.compareBLepton(kLep, kDeltaPhi  ) );
  values.add( "deltaThetaLepBLepton", jetComb.compareBLepton(kLep, kDeltaTheta) );
  // the hadronic b and the neutrino
  values.add( "deltaRHadBNeutrino"    , jetComb.compareBNeutrino(kHad, kDeltaR    ) );
  values.add( "deltaPhiHadBNeutrino"  , jetComb.compareBNeutrino(kHad, kDeltaPhi  ) );
  values.add( "deltaThetaHadBNeutrino", jetComb.compareBNeutrino(kHad, kDeltaTheta) );
  // the leptonic b and the neutrino
  values.add( "deltaRLepBNeutrino"    , jetComb.compareBNeutrino(kLep, kDeltaR    ) );
  values.add( "deltaPhiLepBNeutrino"  , jetComb.compareBNeutrino(kLep, kDeltaPhi  ) );
  values.add( "deltaThetaLepBNeutrino", jetComb.compareBNeutrino(kLep, kDeltaTheta) );
  // ----------------------------------------------------------------------
  // special variables combining the pt of the jets
  // ----------------------------------------------------------------------
  values.add( "relativePtHadronicTop", jetComb.relativePtHadronicTop() );
  values.add( "bOverLightQPt"        , jetComb.bOverLightQPt()         );
  // ----------------------------------------------------------------------
  // variables based on b-tagging with six different algorithms
  // ----------------------------------------------------------------------
  // hadronic b quark candidate
  values.add( "bTagHadBTrkCntHighEff", jetComb.bTag(kHad, kTrkCntHighEff) );
  values.add( "bTagHadBTrkCntHighPur", jetComb.bTag(kHad, kTrkCntHighPur) );
  values.add( "bTagHadBSoftMuon"     , jetComb.bTag(kHad, kSoftMuon     ) );
  values.add( "bTagHadBSimpSecondVtx", jetComb.bTag(kHad, kSimpSecondVtx) );
  values.add( "bTagHadBCombSecondVtx", jetComb.bTag(kHad, kCombSecondVtx) );
  values.add( "bTagHadBImpactParaMVA", jetComb.bTag(kHad, kImpactParaMVA) );
  // leptonic b quark candidate
  values.add( "bTagLepBTrkCntHighEff", jetComb.bTag(kLep, kTrkCntHighEff) );
  values.add( "bTagLepBTrkCntHighPur", jetComb.bTag(kLep, kTrkCntHighPur) );
  values.add( "bTagLepBSoftMuon"     , jetComb.bTag(kLep, kSoftMuon     ) );
  values.add( "bTagLepBSimpSecondVtx", jetComb.bTag(kLep, kSimpSecondVtx) );
  values.add( "bTagLepBCombSecondVtx", jetComb.bTag(kLep, kCombSecondVtx) );
  values.add( "bTagLepBImpactParaMVA", jetComb.bTag(kLep, kImpactParaMVA) );
  // summed b-tags of the two b quark candidates
  values.add( "bTagSumTrkCntHighEff", jetComb.combinedBTags(kTrkCntHighEff, kAdd) );
  values.add( "bTagSumTrkCntHighPur", jetComb.combinedBTags(kTrkCntHighPur, kAdd) );
  values.add( "bTagSumSoftMuon"     , jetComb.combinedBTags(kSoftMuon     , kAdd) );
  values.add( "bTagSumSimpSecondVtx", jetComb.combinedBTags(kSimpSecondVtx, kAdd) );
  values.add( "bTagSumCombSecondVtx", jetComb.combinedBTags(kCombSecondVtx, kAdd) );
  values.add( "bTagSumImpactParaMVA", jetComb.combinedBTags(kImpactParaMVA, kAdd) );
  // multiplied b-tags of the two b quark candidates
  values.add( "bTagProdImpactParaMVA", jetComb.combinedBTags(kImpactParaMVA, kMult) );
}

#endif
