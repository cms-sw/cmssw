#ifndef TtSemiLepSignalSelEval_h
#define TtSemiLepSignalSelEval_h

#include "Math/VectorUtil.h"
#include "TMath.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepSignalSel.h"

inline double evaluateTtSemiLepSignalSel(PhysicsTools::MVAComputerCache& mvaComputer,
				    const TtSemiLepSignalSel& sigsel,
				    const bool training = false, const bool isSignal = false)
{
  std::vector<PhysicsTools::Variable::Value> values;
  
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kTargetId, isSignal) );
  
  values.push_back( PhysicsTools::Variable::Value("dphiMETlepton",     sigsel.dphiMETlepton()     ) );
  values.push_back( PhysicsTools::Variable::Value("dphiMETleadingjet", sigsel.dphiMETleadingjet() ) );
  values.push_back( PhysicsTools::Variable::Value("ETratiojet5jet4",   sigsel.ETratiojet5jet4()   ) );
  values.push_back( PhysicsTools::Variable::Value("aplanarity",        sigsel.aplanar()   ) );
  values.push_back( PhysicsTools::Variable::Value("sphericity",        sigsel.spheric()   ) );
  values.push_back( PhysicsTools::Variable::Value("isotropy",          sigsel.isotrop()   ) );
  values.push_back( PhysicsTools::Variable::Value("circularity",       sigsel.circular()   ) );
  values.push_back( PhysicsTools::Variable::Value("sumEt",             sigsel.sumEt()   ) );
  values.push_back( PhysicsTools::Variable::Value("maxEta",            sigsel.maxEta()   ) );
  values.push_back( PhysicsTools::Variable::Value("Et1",               sigsel.Et1()   ) );
  values.push_back( PhysicsTools::Variable::Value("Et2",               sigsel.Et2()   ) );
  values.push_back( PhysicsTools::Variable::Value("Et3",               sigsel.Et3()   ) );
  values.push_back( PhysicsTools::Variable::Value("Et4",               sigsel.Et4()   ) );
  values.push_back( PhysicsTools::Variable::Value("lepPt",             sigsel.lepPt()   ) );
  
  return mvaComputer->eval( values );
}

#endif
