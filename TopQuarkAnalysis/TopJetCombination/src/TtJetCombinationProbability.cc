#include "TopQuarkAnalysis/TopJetCombination/interface/TtJetCombinationProbability.h"
//
// Constructor
//

TtJetCombinationProbability::TtJetCombinationProbability() {
}

TtJetCombinationProbability::TtJetCombinationProbability(std::string algomodule) {

  std::cout << "=== Constructing a TtJetCombinationProbability... " << std::endl;
  
  // get purity vs. b-discriminant
  combBTagFit = readBTagLR(envUtil("LOCALRT").getEnv()+"/src/TopQuarkAnalysis/TopJetCombination/data/TtSemiBProbLR.root");
  
  // get P(trueCombExist)
  PTrueCombExist = getPExistingTrueComb(envUtil("LOCALRT").getEnv()+"/src/TopQuarkAnalysis/TopJetCombination/data/TtSemiBJetSelectionLR.root");
  
  // get observable fits for BJetSelectionLR
  fBSelObs = getBSelObsFits(envUtil("LOCALRT").getEnv()+"/src/TopQuarkAnalysis/TopJetCombination/data/TtSemiBJetSelectionLR.root");
  
  // get purity vs LRtot for BJetSelectionLR
  fBSelPurity = getBSelPurVsLRtotFit(envUtil("LOCALRT").getEnv()+"/src/TopQuarkAnalysis/TopJetCombination/data/TtSemiBJetSelectionLR.root");
  
  // get observable fits for bHadrSelectionLR
  fBhadrObs = getBhadrObsFits(envUtil("LOCALRT").getEnv()+"/src/TopQuarkAnalysis/TopJetCombination/data/TtSemiBhadrSelectionLR.root");
  
  // get purity vs LRtot for bHadrSelectionLR
  fBhadrPurity = getBhadrPurVsLRtotFit(envUtil("LOCALRT").getEnv()+"/src/TopQuarkAnalysis/TopJetCombination/data/TtSemiBhadrSelectionLR.root");

  std::cout << "=== done." << std::endl;

}


//
// Destructor
//
TtJetCombinationProbability::~TtJetCombinationProbability() {
}






//
// Calculate Jet Combination Probability
//


//calculate P(there exist a correct matching)
double  TtJetCombinationProbability::getPTrueCombExist(TtSemiEvtSolution * sol){
  return PTrueCombExist;
}


//calculate P(correct b-jets|there exist one)
double  TtJetCombinationProbability::getPTrueBJetSel(TtSemiEvtSolution * sol){
  double PcorrectBjets = 0.;
  if(sol->getChi2()>0){
    vector<double> bSelObsVals = getBSelObsValues(sol, &combBTagFit);
    double logLR = 0;
    for (int j = 0; j < nrBSelObs; j++) {
      logLR += log(fBSelObs[j].Eval(bSelObsVals[bSelObs[j]]));
    }
    PcorrectBjets = fBSelPurity.Eval(logLR);
  }
  return PcorrectBjets;
}



//calculate P(correct b-jets|there exist one)
double  TtJetCombinationProbability::getPTrueBhadrSel(TtSemiEvtSolution * sol){
  double PcorrectBhadr = 0.;
  if(sol->getChi2()>0){
    vector<double> bHadrObsVals = getBhadrObsValues(sol, &combBTagFit);
    double logLR = 0;
    for (int j = 0; j < nrBhadrObs; j++) {
      logLR += log(fBhadrObs[j].Eval(bHadrObsVals[bHadrObs[j]]));
    }
    PcorrectBhadr = fBhadrPurity.Eval(logLR);
  }
  return PcorrectBhadr;
}
