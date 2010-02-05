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

/*
  \class   TtSemiLepKinFitter TtSemiLepKinFitter.h "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"
  
  \brief   one line description to be added here...

  text to be added here...
  
**/

class TtSemiLepKinFitter {
  
 public:
  
  /// supported constraints
  enum Constraint { kWHadMass = 1, kWLepMass, kTopHadMass, kTopLepMass, kNeutrinoMass };
  /// supported parameterizations
  enum Param{ kEMom, kEtEtaPhi, kEtThetaPhi };

 public:
  /// default constructor
  explicit TtSemiLepKinFitter();
  /// constructor initialized with built-in types and class enum's custom parameters
  explicit TtSemiLepKinFitter(Param jetParam, Param lepParam, Param metParam, int maxNrIter, double maxDeltaS, double maxF,
			      std::vector<Constraint> constraints, double mW=80.4, double mTop=173.);
  /// default destructor
  ~TtSemiLepKinFitter();

  /// kinematic fit interface
  template <class LeptonType> int fit(const std::vector<pat::Jet>& jets, const pat::Lepton<LeptonType>& leps, const pat::MET& met);
  // return hadronic b quark candidate
  const pat::Particle fittedHadB() const { return (fitter_->getStatus()==0 ? fittedHadB_ : pat::Particle()); };
  // return hadronic light quark candidate
  const pat::Particle fittedHadP() const { return (fitter_->getStatus()==0 ? fittedHadP_ : pat::Particle()); };
  // return hadronic light quark candidate
  const pat::Particle fittedHadQ() const { return (fitter_->getStatus()==0 ? fittedHadQ_ : pat::Particle()); };
  // return leptonic b quark candidate
  const pat::Particle fittedLepB() const { return (fitter_->getStatus()==0 ? fittedLepB_ : pat::Particle()); };
  // return lepton candidate
  const pat::Particle fittedLepton() const { return (fitter_->getStatus()==0 ? fittedLepton_ : pat::Particle()); };
  // return neutrino candidate
  const pat::Particle fittedNeutrino() const { return (fitter_->getStatus()==0 ? fittedNeutrino_ : pat::Particle()); };
  /// return chi2 of fit (not normalized to degrees of freedom)
  double fitS()  const { return fitter_->getS(); };
  /// return number of used iterations
  int fitNrIter() const { return fitter_->getNbIter(); };
  /// return fit probability
  double fitProb() const { return TMath::Prob(fitter_->getS(), fitter_->getNDF()); };
  /// add kin fit information to the old event solution (in for legacy reasons)
  TtSemiEvtSolution addKinFitInfo(TtSemiEvtSolution* asol);
  
 private:
  /// print fitter setup  
  void printSetup() const;
  /// setup fitter  
  void setupFitter();
  /// initialize jet inputs
  void setupJets();
  /// initialize lepton inputs
  void setupLeptons();
  /// initialize constraints
  void setupConstraints();
  /// convert Param to human readable form
  std::string param(const Param& param) const;
  /// change format from TMatrixD to specially sorted vector<float>
  std::vector<float> translateCovM(TMatrixD& inMatrix);
  
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
  /// jet parametrization
  Param  jetParam_;
  /// lepton parametrization
  Param lepParam_;
  /// met parametrization
  Param metParam_;
  /// maximal allowed number of iterations to be used for the fit
  int maxNrIter_;
  /// maximal allowed chi2 (not normalized to degrees of freedom)
  double maxDeltaS_;
  /// maximal allowed distance from constraints
  double maxF_;
  /// vector of constraints to be used
  std::vector<Constraint> constrList_;  
  /// W mass value used for constraints
  double mW_;
  /// top mass value used for constraints
  double mTop_;
};

/// convert Param to human readable form
inline std::string 
TtSemiLepKinFitter::param(const Param& param) const
{
  std::string parName;
  switch(param){
  case kEMom       : parName="EMom";       break;
  case kEtEtaPhi   : parName="EtEtaPhi";   break;
  case kEtThetaPhi : parName="EtThetaPhi"; break;    
  }
  return parName;
}


#endif
