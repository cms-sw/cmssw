#ifndef TtSemiLepKinFitter_h
#define TtSemiLepKinFitter_h

#include <vector>

#include "TLorentzVector.h"

#include "DataFormats/PatCandidates/interface/Lepton.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TopKinFitter.h"

class TAbsFitParticle;
class TFitConstraintM;

/*
  \class   TtSemiLepKinFitter TtSemiLepKinFitter.h "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"
  
  \brief   one line description to be added here...

  text to be added here...
  
**/

class TtSemiLepKinFitter : public TopKinFitter {
  
 public:
  
  /// supported constraints
  enum Constraint { kWHadMass = 1, kWLepMass, kTopHadMass, kTopLepMass, kNeutrinoMass, kEqualTopMasses };

 public:
  /// default constructor
  explicit TtSemiLepKinFitter();
  /// constructor initialized with built-in types and class enum's custom parameters
  explicit TtSemiLepKinFitter(Param jetParam, Param lepParam, Param metParam, int maxNrIter, double maxDeltaS, double maxF,
			      std::vector<Constraint> constraints, double mW=80.4, double mTop=173.);
  /// default destructor
  ~TtSemiLepKinFitter();

  /// kinematic fit interface
  template <class LeptonType> int fit(const std::vector<pat::Jet>& jets, const pat::Lepton<LeptonType>& leps, const pat::MET& met, const double jetResolutionSmearFactor);
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
  /// add kin fit information to the old event solution (in for legacy reasons)
  TtSemiEvtSolution addKinFitInfo(TtSemiEvtSolution* asol, const double jetResolutionSmearFactor=1.);
  
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
  
 private:
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
  /// vector of constraints to be used
  std::vector<Constraint> constrList_;  
};

#endif
