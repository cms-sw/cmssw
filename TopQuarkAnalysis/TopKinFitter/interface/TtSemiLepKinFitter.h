#ifndef TtSemiLepKinFitter_h
#define TtSemiLepKinFitter_h

#include <vector>

#include "TLorentzVector.h"

#include "DataFormats/PatCandidates/interface/Lepton.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TopKinFitter.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/CovarianceMatrix.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TAbsFitParticle;
class TFitConstraintM;
class TFitConstraintEp;

/*
  \class   TtSemiLepKinFitter TtSemiLepKinFitter.h "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"
  
  \brief   one line description to be added here...

  text to be added here...
  
**/

class TtSemiLepKinFitter : public TopKinFitter {
  
 public:
  
  /// supported constraints
  enum Constraint { kWHadMass = 1, kWLepMass, kTopHadMass, kTopLepMass, kNeutrinoMass, kEqualTopMasses, kSumPt };

 public:
  /// default constructor
  explicit TtSemiLepKinFitter();
  /// constructor initialized with built-in types and class enum's custom parameters
  explicit TtSemiLepKinFitter(Param jetParam, Param lepParam, Param metParam, int maxNrIter, double maxDeltaS, double maxF,
			      const std::vector<Constraint>& constraints, double mW=80.4, double mTop=173.,
			      const std::vector<edm::ParameterSet>* udscResolutions=0, 
			      const std::vector<edm::ParameterSet>* bResolutions   =0,
			      const std::vector<edm::ParameterSet>* lepResolutions =0, 
			      const std::vector<edm::ParameterSet>* metResolutions =0,
			      const std::vector<double>* jetEnergyResolutionScaleFactors=0,
			      const std::vector<double>* jetEnergyResolutionEtaBinning  =0);
  /// default destructor
  ~TtSemiLepKinFitter();

  /// kinematic fit interface for PAT objects
  template <class LeptonType> int fit(const std::vector<pat::Jet>& jets, const pat::Lepton<LeptonType>& leps, const pat::MET& met);
  /// kinematic fit interface for plain 4-vecs
  int fit(const TLorentzVector& p4HadP, const TLorentzVector& p4HadQ, const TLorentzVector& p4HadB, const TLorentzVector& p4LepB,
	  const TLorentzVector& p4Lepton, const TLorentzVector& p4Neutrino, const int leptonCharge, const CovarianceMatrix::ObjectType leptonType);
  /// common core of the fit interface
  int fit(const TLorentzVector& p4HadP, const TLorentzVector& p4HadQ, const TLorentzVector& p4HadB, const TLorentzVector& p4LepB,
	  const TLorentzVector& p4Lepton, const TLorentzVector& p4Neutrino,
	  const TMatrixD& covHadP, const TMatrixD& covHadQ, const TMatrixD& covHadB, const TMatrixD& covLepB,
	  const TMatrixD& covLepton, const TMatrixD& covNeutrino,
	  const int leptonCharge);
  /// return hadronic b quark candidate
  const pat::Particle fittedHadB() const { return (fitter_->getStatus()==0 ? fittedHadB_ : pat::Particle()); };
  /// return hadronic light quark candidate
  const pat::Particle fittedHadP() const { return (fitter_->getStatus()==0 ? fittedHadP_ : pat::Particle()); };
  /// return hadronic light quark candidate
  const pat::Particle fittedHadQ() const { return (fitter_->getStatus()==0 ? fittedHadQ_ : pat::Particle()); };
  /// return leptonic b quark candidate
  const pat::Particle fittedLepB() const { return (fitter_->getStatus()==0 ? fittedLepB_ : pat::Particle()); };
  /// return lepton candidate
  const pat::Particle fittedLepton() const { return (fitter_->getStatus()==0 ? fittedLepton_ : pat::Particle()); };
  /// return neutrino candidate
  const pat::Particle fittedNeutrino() const { return (fitter_->getStatus()==0 ? fittedNeutrino_ : pat::Particle()); };
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
  
 private:
  /// input particles
  TAbsFitParticle* hadB_;
  TAbsFitParticle* hadP_;
  TAbsFitParticle* hadQ_;
  TAbsFitParticle* lepB_;
  TAbsFitParticle* lepton_;
  TAbsFitParticle* neutrino_;
  /// resolutions
  const std::vector<edm::ParameterSet>* udscResolutions_;
  const std::vector<edm::ParameterSet>* bResolutions_;
  const std::vector<edm::ParameterSet>* lepResolutions_;
  const std::vector<edm::ParameterSet>* metResolutions_;
  /// scale factors for the jet energy resolution
  const std::vector<double>* jetEnergyResolutionScaleFactors_;
  const std::vector<double>* jetEnergyResolutionEtaBinning_;
  /// object used to construct the covariance matrices for the individual particles
  CovarianceMatrix* covM_;
  /// supported constraints
  std::map<Constraint, TFitConstraintM*> massConstr_;
  TFitConstraintEp* sumPxConstr_;
  TFitConstraintEp* sumPyConstr_;
  /// output particles
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
  /// internally use simple boolean for this constraint to reduce the per-event computing time
  bool constrainSumPt_;
};

template <class LeptonType>
int TtSemiLepKinFitter::fit(const std::vector<pat::Jet>& jets, const pat::Lepton<LeptonType>& lepton, const pat::MET& neutrino)
{
  if( jets.size()<4 )
    throw edm::Exception( edm::errors::Configuration, "Cannot run the TtSemiLepKinFitter with less than 4 jets" );

  // get jets in right order
  const pat::Jet hadP = jets[TtSemiLepEvtPartons::LightQ   ];
  const pat::Jet hadQ = jets[TtSemiLepEvtPartons::LightQBar];
  const pat::Jet hadB = jets[TtSemiLepEvtPartons::HadB     ];
  const pat::Jet lepB = jets[TtSemiLepEvtPartons::LepB     ];
 
  // initialize particles
  const TLorentzVector p4HadP( hadP.px(), hadP.py(), hadP.pz(), hadP.energy() );
  const TLorentzVector p4HadQ( hadQ.px(), hadQ.py(), hadQ.pz(), hadQ.energy() );
  const TLorentzVector p4HadB( hadB.px(), hadB.py(), hadB.pz(), hadB.energy() );
  const TLorentzVector p4LepB( lepB.px(), lepB.py(), lepB.pz(), lepB.energy() );
  const TLorentzVector p4Lepton  ( lepton.px(), lepton.py(), lepton.pz(), lepton.energy() );
  const TLorentzVector p4Neutrino( neutrino.px(), neutrino.py(), 0, neutrino.et() );

  // initialize covariance matrices
  TMatrixD covHadP = covM_->setupMatrix(hadP, jetParam_);
  TMatrixD covHadQ = covM_->setupMatrix(hadQ, jetParam_);
  TMatrixD covHadB = covM_->setupMatrix(hadB, jetParam_, "bjets");
  TMatrixD covLepB = covM_->setupMatrix(lepB, jetParam_, "bjets");
  TMatrixD covLepton   = covM_->setupMatrix(lepton  , lepParam_);
  TMatrixD covNeutrino = covM_->setupMatrix(neutrino, metParam_);

  // now do the part that is fully independent of PAT features
  return fit(p4HadP, p4HadQ, p4HadB, p4LepB, p4Lepton, p4Neutrino,
	     covHadP, covHadQ, covHadB, covLepB, covLepton, covNeutrino,
	     lepton.charge());
}


#endif
