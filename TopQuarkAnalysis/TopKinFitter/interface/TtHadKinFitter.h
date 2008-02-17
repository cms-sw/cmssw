#ifndef TtHadKinFitter_h
#define TtHadKinFitter_h
//
// $Id: TtHadKinFitter.h,v 1.2 2007/10/15 23:51:12 lowette Exp $
// Adapted TtSemiKinFitter.h,v 1.3 2007/09/19 23:08:09 lowette Exp 
// for fully hadronic channel 

#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"

#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"

#include <vector>

class TKinFitter;
class TAbsFitParticle;
class TFitConstraintM;

class TtHadKinFitter {

 public:
  
  enum Parametrization { EMom, EtEtaPhi, EtThetaPhi };
  
 public:
  
  TtHadKinFitter();
  TtHadKinFitter(int jetParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
  TtHadKinFitter(Parametrization jetParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
  ~TtHadKinFitter();
  
  TtHadEvtSolution addKinFitInfo(TtHadEvtSolution * asol);
  
 private:
  
  void setupFitter();
  std::vector<float> translateCovM(TMatrixD &);
  
 private:
  
  // the kinematic fitter
  TKinFitter * theFitter_;
  // the particles that enter the kinematic fit
  TAbsFitParticle * fitHadb_;
  TAbsFitParticle * fitHadp_;
  TAbsFitParticle * fitHadq_;
  TAbsFitParticle * fitHadbbar_;
  TAbsFitParticle * fitHadj_;
  TAbsFitParticle * fitHadk_;
  // the constraints on the fit
  TFitConstraintM  * cons1_;
  TFitConstraintM  * cons2_;
  TFitConstraintM  * cons3_;
  TFitConstraintM  * cons4_;
  TFitConstraintM  * cons5_;
  
  // other parameters
  Parametrization jetParam_;
  int maxNrIter_;
  double maxDeltaS_;
  double maxF_;
  std::vector<int> constraints_;  
};

#endif

