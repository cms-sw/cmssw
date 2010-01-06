#ifndef TtFullHadKinFitter_h
#define TtFullHadKinFitter_h

#include <vector>

#include "TMath.h"
#include "TMatrixD.h"
#include "TLorentzVector.h"

#include "PhysicsTools/KinFitter/interface/TKinFitter.h"

#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"

class TAbsFitParticle;
class TFitConstraintM;

/*
  \class   TtFullHadKinFitter TtFullHadKinFitter.h "TopQuarkAnalysis/TopKinFitter/interface/TtFullHadKinFitter.h"
  
  \brief   one line description to be added here...

  text to be added here...
  
**/

class TtFullHadKinFitter {

 public:
  /// supported constraints
  enum Constraint{ kWPlusMass=1, kWMinusMass, kTopMass, kTopBarMass, kEqualTopMasses };
  /// supported parameterizations
  enum Param{ kEMom, kEtEtaPhi, kEtThetaPhi };
  
 public:
  /// default constructor
  TtFullHadKinFitter();
  /// used to convert vector of int's to vector of constraints (just used in TtFullHadKinFitter(int, int, double, double, std::vector<unsigned int>))
  std::vector<TtFullHadKinFitter::Constraint> intToConstraint(std::vector<unsigned int> constraints);
  /// constructor initialized with build-in types as custom parameters (only included to keep TtHadEvtSolutionMaker.cc running)
  TtFullHadKinFitter(int jetParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<unsigned int> constraints,
		     double mW=80.4, double mTop=173.);
  /// constructor initialized with built-in types and class enum's custom parameters
  TtFullHadKinFitter(Param jetParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<Constraint> constraints,
		     double mW=80.4, double mTop=173.);
  /// default destructor
  ~TtFullHadKinFitter();

  /// kinematic fit interface
  int fit(const std::vector<pat::Jet>& jets);
  /// return fitted b quark candidate
  const pat::Particle fittedB() const { return (fitter_->getStatus()==0 ? fittedB_ : pat::Particle()); };
  /// return fitted b quark candidate
  const pat::Particle fittedBBar() const { return (fitter_->getStatus()==0 ? fittedBBar_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightQ() const { return (fitter_->getStatus()==0 ? fittedLightQ_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightQBar() const { return (fitter_->getStatus()==0 ? fittedLightQBar_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightP() const { return (fitter_->getStatus()==0 ? fittedLightP_ : pat::Particle()); };
  /// return fitted light quark candidate
  const pat::Particle fittedLightPBar() const { return (fitter_->getStatus()==0 ? fittedLightPBar_ : pat::Particle()); };
  /// return chi2 of fit (not normalized to degrees of freedom)
  double fitS()  const { return fitter_->getS(); };
  /// return number of used iterations
  int fitNrIter() const { return fitter_->getNbIter(); };
  /// return fit probability
  double fitProb() const { return TMath::Prob(fitter_->getS(), fitter_->getNDF()); };
  /// add kin fit information to the old event solution (in for legacy reasons)
  TtHadEvtSolution addKinFitInfo(TtHadEvtSolution * asol);
  
 private:
  /// print fitter setup
  void printSetup() const;
  /// setup fitter  
  void setupFitter();
  /// initialize jet inputs
  void setupJets();
  /// initialize constraints
  void setupConstraints();
  /// convert Param to human readable form
  std::string param(const Param& param) const;

 private:
  /// the kinematic fitter
  TKinFitter* fitter_;
  /// input particles
  TAbsFitParticle* b_;
  TAbsFitParticle* bBar_;
  TAbsFitParticle* lightQ_;
  TAbsFitParticle* lightQBar_;
  TAbsFitParticle* lightP_;
  TAbsFitParticle* lightPBar_;
  /// supported constraints
  std::map<Constraint, TFitConstraintM*> massConstr_;
  /// output particles
  pat::Particle fittedB_;
  pat::Particle fittedBBar_;
  pat::Particle fittedLightQ_;
  pat::Particle fittedLightQBar_;
  pat::Particle fittedLightP_;
  pat::Particle fittedLightPBar_;
  /// jet parametrization
  Param jetParam_;
  /// maximal allowed number of iterations to be used for the fit
  int maxNrIter_;
  /// maximal allowed chi2 (not normalized to degrees of freedom)
  double maxDeltaS_;
  /// maximal allowed distance from constraints
  double maxF_;
  /// vector of constraints to be used
  std::vector<Constraint> constraints_;
  /// W mass value used for constraints
  double mW_;
  /// top mass value used for constraints
  double mTop_;
};

/// convert Param to human readable form
inline std::string 
TtFullHadKinFitter::param(const Param& param) const
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

