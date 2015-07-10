#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneAlgorithm.h"
#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneLikelihood.h"

SVfitStandaloneAlgorithm  svFitStandaloneAlgorithm_(const std::vector<MeasuredTauLepton>& measuredTauLeptons, double measuredMETx, double measuredMETy, const TMatrixD& covMET, unsigned int verbose = 0); 
svFitStandalone::MeasuredTauLepton measuredTauLepton_(svFitStandalone::kDecayType type, double pt, double eta, double phi, double mass, int decayMode = -1);
