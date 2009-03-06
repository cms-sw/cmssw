#ifndef TtSemiLepJetCombEval_h
#define TtSemiLepJetCombEval_h

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepJetComb.h"

// ----------------------------------------------------------------------
// common interface for the evaluation of likelihood variables
// for jet combinations in the semi leptonic ttbar decays used
// by the MVATrainer and MVAComputer
// ----------------------------------------------------------------------

inline double evaluateTtSemiLepJetComb(PhysicsTools::MVAComputerCache& mvaComputer, const TtSemiLepJetComb& jetComb, const bool training=false, const bool trueCombi=false)
{
  // namespace for enums
  using namespace JetComb;
  std::vector<PhysicsTools::Variable::Value> values;
  
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kTargetId, trueCombi) );

  // mass of the leptonic top candidate
  values.push_back( PhysicsTools::Variable::Value("massLepTop",            jetComb.topVar(kLep, kMass) ));
  // mass of the hadronic top candidate
  values.push_back( PhysicsTools::Variable::Value("massHadTop",            jetComb.topVar(kHad, kMass) ));
  // mass of the leptonic W candidate
  values.push_back( PhysicsTools::Variable::Value("massLepW",              jetComb.wBosonVar(kLep, kMass) ));  
  // mass of the hadronic W candidate
  values.push_back( PhysicsTools::Variable::Value("massHadW",              jetComb.wBosonVar(kHad, kMass) ));
  // summed btag for trackCountingHighEfficiency tag
  values.push_back( PhysicsTools::Variable::Value("sumBTag1HadBLepB",      jetComb.summedTrackCountingHighEff() ));
  // deltaR bewteen the leptonic b and the lepton candidate
  values.push_back( PhysicsTools::Variable::Value("deltaRLepBLepton",      jetComb.compareBLepton(kLep, kDeltaR) ));
  // deltaTheta between the two lightQ candidates
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadQHadQBar", jetComb.compareLightQuarks(kDeltaTheta) ));
  // deltaTheta between the hadronic W and the hadronic b candidate
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadWHadB",    jetComb.compareWB(kHad, kHad, kDeltaTheta) ));
  // deltaM between the two top candidates
  values.push_back( PhysicsTools::Variable::Value("deltaMassHadTopLepTop", jetComb.compareTopTop(kHad, kLep, kDeltaM) ));
  // deltaTheta between the two top candidates
  values.push_back( PhysicsTools::Variable::Value("deltaThetaHadTopLepTop",jetComb.compareTopTop(kHad, kLep, kDeltaTheta) ));

  return mvaComputer->eval( values );
}

#endif
