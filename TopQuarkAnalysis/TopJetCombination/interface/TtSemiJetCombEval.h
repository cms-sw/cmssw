#ifndef TtSemiJetCombEval_h
#define TtSemiJetCombEval_h

#include "Math/VectorUtil.h"
#include "TMath.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiJetComb.h"

inline double evaluateTtSemiJetComb(PhysicsTools::MVAComputerCache& mvaComputer,
				    const TtSemiJetComb& jetComb,
				    const bool training = false, const bool trueCombi = false)
{
  std::vector<PhysicsTools::Variable::Value> values;
  
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kTargetId, trueCombi) );
  
  values.push_back( PhysicsTools::Variable::Value("angleHadQQBar", jetComb.angleHadQQBar()   ) );
  values.push_back( PhysicsTools::Variable::Value("angleHadWHadB", jetComb.angleHadWHadB()   ) );
  values.push_back( PhysicsTools::Variable::Value("angleMuonLepB", jetComb.angleLeptonLepB() ) );
  values.push_back( PhysicsTools::Variable::Value("angleTopTop",   jetComb.angleTopTop()     ) );
  values.push_back( PhysicsTools::Variable::Value("mHadW",         jetComb.massHadW()        ) );
  values.push_back( PhysicsTools::Variable::Value("mHadTop",       jetComb.massHadTop()      ) );
  values.push_back( PhysicsTools::Variable::Value("deltaMTopTop",  jetComb.deltaMTopTop()    ) );
  
  return mvaComputer->eval( values );
}

#endif
