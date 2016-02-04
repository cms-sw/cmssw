#ifndef TtFullHadSignalSelEval_h
#define TtFullHadSignalSelEval_h

#include "Math/VectorUtil.h"
#include "TMath.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtFullHadSignalSel.h"

inline double evaluateTtFullHadSignalSel(PhysicsTools::MVAComputerCache& mvaComputer,
					 const TtFullHadSignalSel& sigsel, double weight = 1.0,
					 const bool training = false, const bool isSignal = false)
{
  std::vector<PhysicsTools::Variable::Value> values;
  
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kTargetId, isSignal) );
  if(training) values.push_back( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kWeightId, weight));  

  values.push_back( PhysicsTools::Variable::Value("H",      sigsel.H()      ) );
  values.push_back( PhysicsTools::Variable::Value("Ht",     sigsel.Ht()     ) );
  values.push_back( PhysicsTools::Variable::Value("Ht123",  sigsel.Ht123()  ) );
  values.push_back( PhysicsTools::Variable::Value("Ht3jet", sigsel.Ht3jet() ) );
  values.push_back( PhysicsTools::Variable::Value("sqrt_s", sigsel.sqrt_s() ) );
  values.push_back( PhysicsTools::Variable::Value("Et56",   sigsel.Et56()   ) );
  values.push_back( PhysicsTools::Variable::Value("M3",     sigsel.M3()     ) );

  values.push_back( PhysicsTools::Variable::Value("TCHP_Bjets", sigsel.TCHP_Bjets() ) );
  values.push_back( PhysicsTools::Variable::Value("SSV_Bjets",  sigsel.SSV_Bjets()  ) );
  values.push_back( PhysicsTools::Variable::Value("CSV_Bjets",  sigsel.CSV_Bjets()  ) );
  values.push_back( PhysicsTools::Variable::Value("SM_Bjets",   sigsel.SM_Bjets()   ) );

  values.push_back( PhysicsTools::Variable::Value("TCHP_Bjet1", sigsel.TCHP_Bjet1() ) );
  values.push_back( PhysicsTools::Variable::Value("TCHP_Bjet2", sigsel.TCHP_Bjet2() ) );
  values.push_back( PhysicsTools::Variable::Value("TCHP_Bjet3", sigsel.TCHP_Bjet3() ) );
  values.push_back( PhysicsTools::Variable::Value("TCHP_Bjet4", sigsel.TCHP_Bjet4() ) );
  values.push_back( PhysicsTools::Variable::Value("TCHP_Bjet5", sigsel.TCHP_Bjet5() ) );
  values.push_back( PhysicsTools::Variable::Value("TCHP_Bjet6", sigsel.TCHP_Bjet6() ) );
  values.push_back( PhysicsTools::Variable::Value("SSV_Bjet1",  sigsel.SSV_Bjet1()  ) );
  values.push_back( PhysicsTools::Variable::Value("SSV_Bjet2",  sigsel.SSV_Bjet2()  ) );
  values.push_back( PhysicsTools::Variable::Value("SSV_Bjet3",  sigsel.SSV_Bjet3()  ) );
  values.push_back( PhysicsTools::Variable::Value("SSV_Bjet4",  sigsel.SSV_Bjet4()  ) );
  values.push_back( PhysicsTools::Variable::Value("SSV_Bjet5",  sigsel.SSV_Bjet5()  ) );
  values.push_back( PhysicsTools::Variable::Value("SSV_Bjet6",  sigsel.SSV_Bjet6()  ) );
  values.push_back( PhysicsTools::Variable::Value("CSV_Bjet1",  sigsel.CSV_Bjet1()  ) );
  values.push_back( PhysicsTools::Variable::Value("CSV_Bjet2",  sigsel.CSV_Bjet2()  ) );
  values.push_back( PhysicsTools::Variable::Value("CSV_Bjet3",  sigsel.CSV_Bjet3()  ) );
  values.push_back( PhysicsTools::Variable::Value("CSV_Bjet4",  sigsel.CSV_Bjet4()  ) );
  values.push_back( PhysicsTools::Variable::Value("CSV_Bjet5",  sigsel.CSV_Bjet5()  ) );
  values.push_back( PhysicsTools::Variable::Value("CSV_Bjet6",  sigsel.CSV_Bjet6()  ) );
  values.push_back( PhysicsTools::Variable::Value("SM_Bjet1",   sigsel.SM_Bjet1()   ) );
  values.push_back( PhysicsTools::Variable::Value("SM_Bjet2",   sigsel.SM_Bjet2()   ) );
  values.push_back( PhysicsTools::Variable::Value("SM_Bjet3",   sigsel.SM_Bjet3()   ) );
  values.push_back( PhysicsTools::Variable::Value("SM_Bjet4",   sigsel.SM_Bjet4()   ) );
  values.push_back( PhysicsTools::Variable::Value("SM_Bjet5",   sigsel.SM_Bjet5()   ) );
  values.push_back( PhysicsTools::Variable::Value("SM_Bjet6",   sigsel.SM_Bjet6()   ) );

  values.push_back( PhysicsTools::Variable::Value("pt1", sigsel.pt1() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2", sigsel.pt2() ) );
  values.push_back( PhysicsTools::Variable::Value("pt3", sigsel.pt3() ) );
  values.push_back( PhysicsTools::Variable::Value("pt4", sigsel.pt4() ) );
  values.push_back( PhysicsTools::Variable::Value("pt5", sigsel.pt5() ) );
  values.push_back( PhysicsTools::Variable::Value("pt6", sigsel.pt6() ) );

  values.push_back( PhysicsTools::Variable::Value("pt1_pt2", sigsel.pt1_pt2() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt3", sigsel.pt1_pt3() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt4", sigsel.pt1_pt4() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt5", sigsel.pt1_pt5() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt6", sigsel.pt1_pt6() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt3", sigsel.pt2_pt3() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt4", sigsel.pt2_pt4() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt5", sigsel.pt2_pt5() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt6", sigsel.pt2_pt6() ) );
  values.push_back( PhysicsTools::Variable::Value("pt3_pt4", sigsel.pt3_pt4() ) );
  values.push_back( PhysicsTools::Variable::Value("pt3_pt5", sigsel.pt3_pt5() ) );
  values.push_back( PhysicsTools::Variable::Value("pt3_pt6", sigsel.pt3_pt6() ) );
  values.push_back( PhysicsTools::Variable::Value("pt4_pt5", sigsel.pt4_pt5() ) );
  values.push_back( PhysicsTools::Variable::Value("pt4_pt6", sigsel.pt4_pt6() ) );
  values.push_back( PhysicsTools::Variable::Value("pt5_pt6", sigsel.pt5_pt6() ) );

  values.push_back( PhysicsTools::Variable::Value("pt1_pt2_norm", sigsel.pt1_pt2_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt3_norm", sigsel.pt1_pt3_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt4_norm", sigsel.pt1_pt4_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt5_norm", sigsel.pt1_pt5_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt1_pt6_norm", sigsel.pt1_pt6_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt3_norm", sigsel.pt2_pt3_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt4_norm", sigsel.pt2_pt4_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt5_norm", sigsel.pt2_pt5_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt2_pt6_norm", sigsel.pt2_pt6_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt3_pt4_norm", sigsel.pt3_pt4_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt3_pt5_norm", sigsel.pt3_pt5_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt3_pt6_norm", sigsel.pt3_pt6_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt4_pt5_norm", sigsel.pt4_pt5_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt4_pt6_norm", sigsel.pt4_pt6_norm() ) );
  values.push_back( PhysicsTools::Variable::Value("pt5_pt6_norm", sigsel.pt5_pt6_norm() ) );

  values.push_back( PhysicsTools::Variable::Value("jet1_etaetaMoment", sigsel.jet1_etaetaMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet2_etaetaMoment", sigsel.jet2_etaetaMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet3_etaetaMoment", sigsel.jet3_etaetaMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet4_etaetaMoment", sigsel.jet4_etaetaMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet5_etaetaMoment", sigsel.jet5_etaetaMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet6_etaetaMoment", sigsel.jet6_etaetaMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet1_etaphiMoment", sigsel.jet1_etaphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet2_etaphiMoment", sigsel.jet2_etaphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet3_etaphiMoment", sigsel.jet3_etaphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet4_etaphiMoment", sigsel.jet4_etaphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet5_etaphiMoment", sigsel.jet5_etaphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet6_etaphiMoment", sigsel.jet6_etaphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet1_phiphiMoment", sigsel.jet1_phiphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet2_phiphiMoment", sigsel.jet2_phiphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet3_phiphiMoment", sigsel.jet3_phiphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet4_phiphiMoment", sigsel.jet4_phiphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet5_phiphiMoment", sigsel.jet5_phiphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jet6_phiphiMoment", sigsel.jet6_phiphiMoment() ) );

  values.push_back( PhysicsTools::Variable::Value("jets_etaetaMoment", sigsel.jets_etaetaMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jets_etaphiMoment", sigsel.jets_etaphiMoment() ) );
  values.push_back( PhysicsTools::Variable::Value("jets_phiphiMoment", sigsel.jets_phiphiMoment() ) );

  values.push_back( PhysicsTools::Variable::Value("aplanarity",  sigsel.aplanarity()  ) );
  values.push_back( PhysicsTools::Variable::Value("sphericity",  sigsel.sphericity()  ) );
  values.push_back( PhysicsTools::Variable::Value("circularity", sigsel.circularity() ) );
  values.push_back( PhysicsTools::Variable::Value("isotropy",    sigsel.isotropy()    ) );
  values.push_back( PhysicsTools::Variable::Value("C",           sigsel.C()           ) );
  values.push_back( PhysicsTools::Variable::Value("D",           sigsel.D()           ) );
  values.push_back( PhysicsTools::Variable::Value("centrality",  sigsel.centrality()  ) );

  values.push_back( PhysicsTools::Variable::Value("dRMin1",            sigsel.dRMin1()        ) );
  values.push_back( PhysicsTools::Variable::Value("dRMin2",            sigsel.dRMin2()        ) );
  values.push_back( PhysicsTools::Variable::Value("sumDR3JetMin1",     sigsel.sumDR3JetMin1() ) );
  values.push_back( PhysicsTools::Variable::Value("sumDR3JetMin2",     sigsel.sumDR3JetMin2() ) );

  values.push_back( PhysicsTools::Variable::Value("dRMin1Mass",        sigsel.dRMin1Mass()        ) );
  values.push_back( PhysicsTools::Variable::Value("dRMin2Mass",        sigsel.dRMin2Mass()        ) );
  values.push_back( PhysicsTools::Variable::Value("sumDR3JetMin1Mass", sigsel.sumDR3JetMin1Mass() ) );
  values.push_back( PhysicsTools::Variable::Value("sumDR3JetMin2Mass", sigsel.sumDR3JetMin2Mass() ) );

  return mvaComputer->eval( values );
    
}

#endif
