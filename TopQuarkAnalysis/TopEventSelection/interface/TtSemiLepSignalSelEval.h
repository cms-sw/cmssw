#ifndef TtSemiLepSignalSelEval_h
#define TtSemiLepSignalSelEval_h

#include "Math/VectorUtil.h"
#include "TMath.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLepSignalSel.h"

inline double evaluateTtSemiLepSignalSel(PhysicsTools::MVAComputerCache& mvaComputer,
					 const TtSemiLepSignalSel& sigsel, float weight = 1.,
					 const bool training = false, const bool isSignal = false)
{
  std::vector<PhysicsTools::Variable::Value> values;
  
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kTargetId, isSignal) );
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kWeightId, weight));  

  values.push_back( PhysicsTools::Variable::Value("sumEt",             sigsel.sumEt()   ) ); 
  values.push_back( PhysicsTools::Variable::Value("relEt1",            sigsel.Et1()   ) ); 
  values.push_back( PhysicsTools::Variable::Value("Abs_lepeta",        sigsel.lepeta()   ) ); 
  values.push_back( PhysicsTools::Variable::Value("MET",               sigsel.MET() ) ); 
  
  values.push_back( PhysicsTools::Variable::Value("dphiMETlepton",     sigsel.dphiMETlepton()     ) );
    
  values.push_back( PhysicsTools::Variable::Value("detajet2jet3",     sigsel.detajet2jet3()     ) ); 
  values.push_back( PhysicsTools::Variable::Value("detajet3jet4",     sigsel.detajet3jet4()     ) ); 

  values.push_back( PhysicsTools::Variable::Value("mindijetmass",     sigsel.mindijetmass()     ) ); 
  values.push_back( PhysicsTools::Variable::Value("maxdijetmass",     sigsel.maxdijetmass()     ) ); 

  values.push_back( PhysicsTools::Variable::Value("mindRjetlepton",     sigsel.mindRjetlepton()     ) );
  

  return mvaComputer->eval( values );
    
}

#endif
