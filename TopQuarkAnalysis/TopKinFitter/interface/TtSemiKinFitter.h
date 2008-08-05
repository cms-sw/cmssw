//
// $Id: TtSemiKinFitter.h,v 1.4.2.2 2008/08/04 09:00:00 snaumann Exp $
//

#ifndef TopKinFitter_TtSemiKinFitter_h
#define TopKinFitter_TtSemiKinFitter_h

#include "PhysicsTools/KinFitter/interface/TKinFitter.h"

#include "DataFormats/PatCandidates/interface/Lepton.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"

#include <vector>

class TAbsFitParticle;
class TFitConstraintM;

class TtSemiKinFitter {
  
 public:
  
  enum Parametrization { EMom, EtEtaPhi, EtThetaPhi };
  
 public:
  
  TtSemiKinFitter();
  TtSemiKinFitter(int jetParam, int lepParam, int metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
  TtSemiKinFitter(Parametrization jetParam, Parametrization lepParam, Parametrization metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
  ~TtSemiKinFitter();

  template <class LeptonType>
  int fit(const std::vector<pat::Jet>&, const pat::Lepton<LeptonType>&, const pat::MET&);

  const pat::Particle getFitHadb() const { return (theFitter_->getStatus()==0 ? aFitHadb_ : pat::Particle()); };
  const pat::Particle getFitHadp() const { return (theFitter_->getStatus()==0 ? aFitHadp_ : pat::Particle()); };
  const pat::Particle getFitHadq() const { return (theFitter_->getStatus()==0 ? aFitHadq_ : pat::Particle()); };
  const pat::Particle getFitLepb() const { return (theFitter_->getStatus()==0 ? aFitLepb_ : pat::Particle()); };
  const pat::Particle getFitLepl() const { return (theFitter_->getStatus()==0 ? aFitLepl_ : pat::Particle()); };
  const pat::Particle getFitLepn() const { return (theFitter_->getStatus()==0 ? aFitLepn_ : pat::Particle()); };

  int getNIter() const { return theFitter_->getNbIter(); };
  double getS()  const { return theFitter_->getS(); };
  double getProb() const { return TMath::Prob(theFitter_->getS(), theFitter_->getNDF()); };
  
  TtSemiEvtSolution addKinFitInfo(TtSemiEvtSolution * asol);
  
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
  TAbsFitParticle * fitLepb_;
  TAbsFitParticle * fitLepl_;
  TAbsFitParticle * fitLepn_;
  // the constraints on the fit
  TFitConstraintM  * cons1_;
  TFitConstraintM  * cons2_;
  TFitConstraintM  * cons3_;
  TFitConstraintM  * cons4_;
  TFitConstraintM  * cons5_;
  // the particles that result from the kinematic fit
  pat::Particle aFitHadb_;
  pat::Particle aFitHadp_;
  pat::Particle aFitHadq_;
  pat::Particle aFitLepb_;
  pat::Particle aFitLepl_;
  pat::Particle aFitLepn_;
  // other parameters
  Parametrization jetParam_, lepParam_, metParam_;
  bool doNeutrinoResol_;
  int maxNrIter_;
  double maxDeltaS_;
  double maxF_;
  std::vector<int> constraints_;  
};

#endif
