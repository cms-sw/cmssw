#ifndef TtSemiLepKinFitter_h
#define TtSemiLepKinFitter_h

#include <vector>

#include "TMath.h"
#include "TMatrixD.h"
#include "TLorentzVector.h"

#include "DataFormats/PatCandidates/interface/Lepton.h"
#include "PhysicsTools/KinFitter/interface/TKinFitter.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

class TAbsFitParticle;
class TFitConstraintM;


class TtSemiLepKinFitter {
  
 public:
  
  // supported constraints
  enum Constraint 
    { kWHadMass = 1, 
      kWLepMass, 
      kTopHadMass, 
      kTopLepMass, 
      kNeutrinoMass
    };
  // supported parameterizations
  enum Param
    { kEMom, 
      kEtEtaPhi, 
      kEtThetaPhi 
    };

 public:
  
  // default contructor
  explicit TtSemiLepKinFitter();
  // with custom parameters
  explicit TtSemiLepKinFitter(Param  jetParam, 
			      Param  lepParam, 
			      Param  metParam, 
			      int    maxNrIter, 
			      double maxDeltaS, 
			      double maxF, 
			      std::vector<Constraint> constraints);
  // destructor
  ~TtSemiLepKinFitter();

  // worker function
  template <class LeptonType> int fit(const std::vector<pat::Jet>  & jets, const pat::Lepton<LeptonType>& leps, const pat::MET& met);

  // return functions for fitted particles
  const pat::Particle fittedHadB()     const { return (fitter_->getStatus()==0 ? fittedHadB_     : pat::Particle()); };
  const pat::Particle fittedHadP()     const { return (fitter_->getStatus()==0 ? fittedHadP_     : pat::Particle()); };
  const pat::Particle fittedHadQ()     const { return (fitter_->getStatus()==0 ? fittedHadQ_     : pat::Particle()); };
  const pat::Particle fittedLepB()     const { return (fitter_->getStatus()==0 ? fittedLepB_     : pat::Particle()); };
  const pat::Particle fittedLepton()   const { return (fitter_->getStatus()==0 ? fittedLepton_   : pat::Particle()); };
  const pat::Particle fittedNeutrino() const { return (fitter_->getStatus()==0 ? fittedNeutrino_ : pat::Particle()); };

  // get functions for fit meta information
  double fitS()  const { return fitter_->getS(); };
  int fitNrIter() const { return fitter_->getNbIter(); };
  double fitProb() const { return TMath::Prob(fitter_->getS(), fitter_->getNDF()); };

  // add kin fit information to the old event solution
  TtSemiEvtSolution addKinFitInfo(TtSemiEvtSolution* asol);
  
 private:
  
  void printSetup();
  void setupFitter();
  void setupJets();
  void setupLeptons();
  void setupConstraints();
  // convert Param to human readable form
  std::string param(Param& param){
    std::string parName;
    switch(param){
    case kEMom       : parName="EMom";       break;
    case kEtEtaPhi   : parName="EtEtaPhi";   break;
    case kEtThetaPhi : parName="EtThetaPhi"; break;    
    }
    return parName;
  }
  std::vector<float> translateCovM(TMatrixD &);
  
 private:

  // kinematic fitter
  TKinFitter* fitter_;
  
  // input particles
  TAbsFitParticle* hadB_;
  TAbsFitParticle* hadP_;
  TAbsFitParticle* hadQ_;
  TAbsFitParticle* lepB_;
  TAbsFitParticle* lepton_;
  TAbsFitParticle* neutrino_;

  // supported constraints
  std::map<Constraint, TFitConstraintM*> massConstr_;

  // output particles
  pat::Particle fittedHadB_;
  pat::Particle fittedHadP_;
  pat::Particle fittedHadQ_;
  pat::Particle fittedLepB_;
  pat::Particle fittedLepton_;
  pat::Particle fittedNeutrino_;

  // steerables
  Param  jetParam_, lepParam_, metParam_;
  int    maxNrIter_;
  double maxDeltaS_;
  double maxF_;
  std::vector<Constraint> constrList_;  
};

#endif
