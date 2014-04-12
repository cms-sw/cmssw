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
  // ----------------------------------------------------------------------
  // mass, pt, eta, phi and theta of single candidates of the ttbar system
  // ----------------------------------------------------------------------
  // hadronic top quark
  values.add( "massHadTop" , jetComb.topVar(JetComb::kHad, JetComb::kMass ) );
  values.add( "ptHadTop"   , jetComb.topVar(JetComb::kHad, JetComb::kPt   ) );
  values.add( "etaHadTop"  , jetComb.topVar(JetComb::kHad, JetComb::kEta  ) );
  values.add( "phiHadTop"  , jetComb.topVar(JetComb::kHad, JetComb::kPhi  ) );
  values.add( "thetaHadTop", jetComb.topVar(JetComb::kHad, JetComb::kTheta) );
  // leptonic top quark
  values.add( "massLepTop" , jetComb.topVar(JetComb::kLep, JetComb::kMass ) );
  values.add( "ptLepTop"   , jetComb.topVar(JetComb::kLep, JetComb::kPt   ) );
  values.add( "etaLepTop"  , jetComb.topVar(JetComb::kLep, JetComb::kEta  ) );
  values.add( "phiLepTop"  , jetComb.topVar(JetComb::kLep, JetComb::kPhi  ) );
  values.add( "thetaLepTop", jetComb.topVar(JetComb::kLep, JetComb::kTheta) );
  // hadronic W boson
  values.add( "massHadW" , jetComb.wBosonVar(JetComb::kHad, JetComb::kMass ) );
  values.add( "ptHadW"   , jetComb.wBosonVar(JetComb::kHad, JetComb::kPt   ) );
  values.add( "etaHadW"  , jetComb.wBosonVar(JetComb::kHad, JetComb::kEta  ) );
  values.add( "phiHadW"  , jetComb.wBosonVar(JetComb::kHad, JetComb::kPhi  ) );
  values.add( "thetaHadW", jetComb.wBosonVar(JetComb::kHad, JetComb::kTheta) );
  // hadronic b quark
  values.add( "ptHadB"   , jetComb.bQuarkVar(JetComb::kHad, JetComb::kPt   ) );
  values.add( "etaHadB"  , jetComb.bQuarkVar(JetComb::kHad, JetComb::kEta  ) );
  values.add( "phiHadB"  , jetComb.bQuarkVar(JetComb::kHad, JetComb::kPhi  ) );
  values.add( "thetaHadB", jetComb.bQuarkVar(JetComb::kHad, JetComb::kTheta) );
  // leptonic b quark
  values.add( "ptLepB"   , jetComb.bQuarkVar(JetComb::kLep, JetComb::kPt   ) );
  values.add( "etaLepB"  , jetComb.bQuarkVar(JetComb::kLep, JetComb::kEta  ) );
  values.add( "phiLepB"  , jetComb.bQuarkVar(JetComb::kLep, JetComb::kPhi  ) );
  values.add( "thetaLepB", jetComb.bQuarkVar(JetComb::kLep, JetComb::kTheta) );
  // light quark
  values.add( "ptLightQ"   , jetComb.lightQVar(false, JetComb::kPt   ) );
  values.add( "etaLightQ"  , jetComb.lightQVar(false, JetComb::kEta  ) );
  values.add( "phiLightQ"  , jetComb.lightQVar(false, JetComb::kPhi  ) );
  values.add( "thetaLightQ", jetComb.lightQVar(false, JetComb::kTheta) );
  // light anti-quark
  values.add( "ptLightQBar"   , jetComb.lightQVar(true, JetComb::kPt   ) );
  values.add( "etaLightQBar"  , jetComb.lightQVar(true, JetComb::kEta  ) );
  values.add( "phiLightQBar"  , jetComb.lightQVar(true, JetComb::kPhi  ) );
  values.add( "thetaLightQBar", jetComb.lightQVar(true, JetComb::kTheta) );
  // ----------------------------------------------------------------------
  // compare two candidates of the ttbar system in DeltaM, DeltaR, DeltaPhi or DeltaTheta
  // ----------------------------------------------------------------------
  // the two top quarks
  values.add( "deltaMHadTopLepTop"    , jetComb.compareHadTopLepTop(JetComb::kDeltaM    ) );
  values.add( "deltaRHadTopLepTop"    , jetComb.compareHadTopLepTop(JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadTopLepTop"  , jetComb.compareHadTopLepTop(JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepTop", jetComb.compareHadTopLepTop(JetComb::kDeltaTheta) );
  // the two W bosons
  values.add( "deltaMHadWLepW"    , jetComb.compareHadWLepW(JetComb::kDeltaM    ) );
  values.add( "deltaRHadWLepW"    , jetComb.compareHadWLepW(JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadWLepW"  , jetComb.compareHadWLepW(JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadWLepW", jetComb.compareHadWLepW(JetComb::kDeltaTheta) );
  // the two b quarks
  values.add( "deltaRHadBLepB"    , jetComb.compareHadBLepB(JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadBLepB"  , jetComb.compareHadBLepB(JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadBLepB", jetComb.compareHadBLepB(JetComb::kDeltaTheta) );
  // the two light quarks
  values.add( "deltaRHadQHadQBar"    , jetComb.compareLightQuarks(JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadQHadQBar"  , jetComb.compareLightQuarks(JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadQHadQBar", jetComb.compareLightQuarks(JetComb::kDeltaTheta) );
  // the hadronic top and the hadronic W
  values.add( "deltaMHadTopHadW"    , jetComb.compareTopW(JetComb::kHad, JetComb::kHad, JetComb::kDeltaM    ) );
  values.add( "deltaRHadTopHadW"    , jetComb.compareTopW(JetComb::kHad, JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadTopHadW"  , jetComb.compareTopW(JetComb::kHad, JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadTopHadW", jetComb.compareTopW(JetComb::kHad, JetComb::kHad, JetComb::kDeltaTheta) );
  // the leptonic top and the leptonic W
  values.add( "deltaMLepTopLepW"    , jetComb.compareTopW(JetComb::kLep, JetComb::kLep, JetComb::kDeltaM    ) );
  values.add( "deltaRLepTopLepW"    , jetComb.compareTopW(JetComb::kLep, JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepTopLepW"  , jetComb.compareTopW(JetComb::kLep, JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepTopLepW", jetComb.compareTopW(JetComb::kLep, JetComb::kLep, JetComb::kDeltaTheta) );
  // the hadronic top and the leptonic W
  values.add( "deltaMHadTopLepW"    , jetComb.compareTopW(JetComb::kHad, JetComb::kLep, JetComb::kDeltaM    ) );
  values.add( "deltaRHadTopLepW"    , jetComb.compareTopW(JetComb::kHad, JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadTopLepW"  , jetComb.compareTopW(JetComb::kHad, JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepW", jetComb.compareTopW(JetComb::kHad, JetComb::kLep, JetComb::kDeltaTheta) );
  // the leptonic top and the hadronic W
  values.add( "deltaMLepTopHadW"    , jetComb.compareTopW(JetComb::kLep, JetComb::kHad, JetComb::kDeltaM    ) );
  values.add( "deltaRLepTopHadW"    , jetComb.compareTopW(JetComb::kLep, JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepTopHadW"  , jetComb.compareTopW(JetComb::kLep, JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepTopHadW", jetComb.compareTopW(JetComb::kLep, JetComb::kHad, JetComb::kDeltaTheta) );
  // the hadronic top and the hadronic b
  values.add( "deltaRHadTopHadB"    , jetComb.compareTopB(JetComb::kHad, JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadTopHadB"  , jetComb.compareTopB(JetComb::kHad, JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadTopHadB", jetComb.compareTopB(JetComb::kHad, JetComb::kHad, JetComb::kDeltaTheta) );
  // the leptonic top and the leptonic b
  values.add( "deltaRLepTopLepB"    , jetComb.compareTopB(JetComb::kLep, JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepTopLepB"  , jetComb.compareTopB(JetComb::kLep, JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepTopLepB", jetComb.compareTopB(JetComb::kLep, JetComb::kLep, JetComb::kDeltaTheta) );
  // the hadronic top and the leptonic b
  values.add( "deltaRHadTopLepB"    , jetComb.compareTopB(JetComb::kHad, JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadTopLepB"  , jetComb.compareTopB(JetComb::kHad, JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepB", jetComb.compareTopB(JetComb::kHad, JetComb::kLep, JetComb::kDeltaTheta) );
  // the leptonic top and the hadronic b
  values.add( "deltaRLepTopHadB"    , jetComb.compareTopB(JetComb::kLep, JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepTopHadB"  , jetComb.compareTopB(JetComb::kLep, JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepTopHadB", jetComb.compareTopB(JetComb::kLep, JetComb::kHad, JetComb::kDeltaTheta) );
  // the hadronic W and the hadronic b
  values.add( "deltaRHadWHadB"    , jetComb.compareWB(JetComb::kHad, JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadWHadB"  , jetComb.compareWB(JetComb::kHad, JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadWHadB", jetComb.compareWB(JetComb::kHad, JetComb::kHad, JetComb::kDeltaTheta) );
  // the leptonic W and the leptonic b
  values.add( "deltaRLepWLepB"    , jetComb.compareWB(JetComb::kLep, JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepWLepB"  , jetComb.compareWB(JetComb::kLep, JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepWLepB", jetComb.compareWB(JetComb::kLep, JetComb::kLep, JetComb::kDeltaTheta) );
  // the hadronic W and the leptonic b
  values.add( "deltaRHadWLepB"    , jetComb.compareWB(JetComb::kHad, JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadWLepB"  , jetComb.compareWB(JetComb::kHad, JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadWLepB", jetComb.compareWB(JetComb::kHad, JetComb::kLep, JetComb::kDeltaTheta) );
  // the leptonic W and the hadronic b
  values.add( "deltaRLepWHadB"    , jetComb.compareWB(JetComb::kLep, JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepWHadB"  , jetComb.compareWB(JetComb::kLep, JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepWHadB", jetComb.compareWB(JetComb::kLep, JetComb::kHad, JetComb::kDeltaTheta) );
  // the hadronic top and the lepton
  values.add( "deltaRHadTopLepton"    , jetComb.compareTopLepton(JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadTopLepton"  , jetComb.compareTopLepton(JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadTopLepton", jetComb.compareTopLepton(JetComb::kHad, JetComb::kDeltaTheta) );
  // the leptonic top and the lepton
  values.add( "deltaRLepTopLepton"    , jetComb.compareTopLepton(JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepTopLepton"  , jetComb.compareTopLepton(JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepTopLepton", jetComb.compareTopLepton(JetComb::kLep, JetComb::kDeltaTheta) );
  // the hadronic top and the neutrino
  values.add( "deltaRHadTopNeutrino"    , jetComb.compareTopNeutrino(JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadTopNeutrino"  , jetComb.compareTopNeutrino(JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadTopNeutrino", jetComb.compareTopNeutrino(JetComb::kHad, JetComb::kDeltaTheta) );
  // the leptonic top and the neutrino
  values.add( "deltaRLepTopNeutrino"    , jetComb.compareTopNeutrino(JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepTopNeutrino"  , jetComb.compareTopNeutrino(JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepTopNeutrino", jetComb.compareTopNeutrino(JetComb::kLep, JetComb::kDeltaTheta) );
  // the hadronic W and the lepton
  values.add( "deltaRHadWLepton"    , jetComb.compareWLepton(JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadWLepton"  , jetComb.compareWLepton(JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadWLepton", jetComb.compareWLepton(JetComb::kHad, JetComb::kDeltaTheta) );
  // the hadronic W and the neutrino
  values.add( "deltaRHadWNeutrino"    , jetComb.compareWNeutrino(JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadWNeutrino"  , jetComb.compareWNeutrino(JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadWNeutrino", jetComb.compareWNeutrino(JetComb::kHad, JetComb::kDeltaTheta) );
  // the hadronic b and the lepton
  values.add( "deltaRHadBLepton"    , jetComb.compareBLepton(JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadBLepton"  , jetComb.compareBLepton(JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadBLepton", jetComb.compareBLepton(JetComb::kHad, JetComb::kDeltaTheta) );
  // the leptonic b and the lepton
  values.add( "deltaRLepBLepton"    , jetComb.compareBLepton(JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepBLepton"  , jetComb.compareBLepton(JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepBLepton", jetComb.compareBLepton(JetComb::kLep, JetComb::kDeltaTheta) );
  // the hadronic b and the neutrino
  values.add( "deltaRHadBNeutrino"    , jetComb.compareBNeutrino(JetComb::kHad, JetComb::kDeltaR    ) );
  values.add( "deltaPhiHadBNeutrino"  , jetComb.compareBNeutrino(JetComb::kHad, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaHadBNeutrino", jetComb.compareBNeutrino(JetComb::kHad, JetComb::kDeltaTheta) );
  // the leptonic b and the neutrino
  values.add( "deltaRLepBNeutrino"    , jetComb.compareBNeutrino(JetComb::kLep, JetComb::kDeltaR    ) );
  values.add( "deltaPhiLepBNeutrino"  , jetComb.compareBNeutrino(JetComb::kLep, JetComb::kDeltaPhi  ) );
  values.add( "deltaThetaLepBNeutrino", jetComb.compareBNeutrino(JetComb::kLep, JetComb::kDeltaTheta) );
  // ----------------------------------------------------------------------
  // special variables combining the pt of the jets
  // ----------------------------------------------------------------------
  values.add( "relativePtHadronicTop", jetComb.relativePtHadronicTop() );
  values.add( "bOverLightQPt"        , jetComb.bOverLightQPt()         );
  // ----------------------------------------------------------------------
  // variables based on b-tagging with six different algorithms
  // ----------------------------------------------------------------------
  // hadronic b quark candidate
  values.add( "bTagHadBTrkCntHighEff", jetComb.bTag(JetComb::kHad, JetComb::kTrackCountHighEff) );
  values.add( "bTagHadBTrkCntHighPur", jetComb.bTag(JetComb::kHad, JetComb::kTrackCountHighPur) );
  values.add( "bTagHadBSoftMuon"     , jetComb.bTag(JetComb::kHad, JetComb::kSoftMuon         ) );
  values.add( "bTagHadBSimpSecondVtx", jetComb.bTag(JetComb::kHad, JetComb::kSimpleSecondVtx  ) );
  values.add( "bTagHadBCombSecondVtx", jetComb.bTag(JetComb::kHad, JetComb::kCombSecondVtx    ) );
  values.add( "bTagHadBImpactParaMVA", jetComb.bTag(JetComb::kHad, JetComb::kCombSecondVtxMVA ) );
  // leptonic b quark candidate
  values.add( "bTagLepBTrkCntHighEff", jetComb.bTag(JetComb::kLep, JetComb::kTrackCountHighEff) );
  values.add( "bTagLepBTrkCntHighPur", jetComb.bTag(JetComb::kLep, JetComb::kTrackCountHighPur) );
  values.add( "bTagLepBSoftMuon"     , jetComb.bTag(JetComb::kLep, JetComb::kSoftMuon         ) );
  values.add( "bTagLepBSimpSecondVtx", jetComb.bTag(JetComb::kLep, JetComb::kSimpleSecondVtx  ) );
  values.add( "bTagLepBCombSecondVtx", jetComb.bTag(JetComb::kLep, JetComb::kCombSecondVtx    ) );
  values.add( "bTagLepBImpactParaMVA", jetComb.bTag(JetComb::kLep, JetComb::kCombSecondVtxMVA ) );
  // summed b-tags of the two b quark candidates
  values.add( "bTagSumTrkCntHighEff", jetComb.combinedBTags(JetComb::kTrackCountHighEff, JetComb::kAdd) );
  values.add( "bTagSumTrkCntHighPur", jetComb.combinedBTags(JetComb::kTrackCountHighPur, JetComb::kAdd) );
  values.add( "bTagSumSoftMuon"     , jetComb.combinedBTags(JetComb::kSoftMuon,          JetComb::kAdd) );
  values.add( "bTagSumSimpSecondVtx", jetComb.combinedBTags(JetComb::kSimpleSecondVtx,   JetComb::kAdd) );
  values.add( "bTagSumCombSecondVtx", jetComb.combinedBTags(JetComb::kCombSecondVtx,     JetComb::kAdd) );
  values.add( "bTagSumImpactParaMVA", jetComb.combinedBTags(JetComb::kCombSecondVtxMVA,  JetComb::kAdd) );
  // multiplied b-tags of the two b quark candidates
  values.add( "bTagProdImpactParaMVA", jetComb.combinedBTags(JetComb::kCombSecondVtxMVA, JetComb::kMult) );
  // summed b-tags of the two light quark candidates
  values.add( "bTagSumHadQHadQBarTrkCntHighEff", jetComb.combinedBTagsForLightQuarks(JetComb::kTrackCountHighEff, JetComb::kAdd) );
  values.add( "bTagSumHadQHadQBarTrkCntHighPur", jetComb.combinedBTagsForLightQuarks(JetComb::kTrackCountHighPur, JetComb::kAdd) );
  values.add( "bTagSumHadQHadQBarSoftMuon"     , jetComb.combinedBTagsForLightQuarks(JetComb::kSoftMuon,          JetComb::kAdd) );
  values.add( "bTagSumHadQHadQBarSimpSecondVtx", jetComb.combinedBTagsForLightQuarks(JetComb::kSimpleSecondVtx,   JetComb::kAdd) );
  values.add( "bTagSumHadQHadQBarCombSecondVtx", jetComb.combinedBTagsForLightQuarks(JetComb::kCombSecondVtx,     JetComb::kAdd) );
  values.add( "bTagSumHadQHadQBarImpactParaMVA", jetComb.combinedBTagsForLightQuarks(JetComb::kCombSecondVtxMVA,  JetComb::kAdd) );
  // multiplied b-tags of the two light quark candidates
  values.add( "bTagProdHadQHadQBarImpactParaMVA", jetComb.combinedBTagsForLightQuarks(JetComb::kCombSecondVtxMVA, JetComb::kMult) );
}

#endif
