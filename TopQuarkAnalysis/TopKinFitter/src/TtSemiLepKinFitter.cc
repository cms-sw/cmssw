#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// default configuration is: Parametrization kEMom, Max iterations = 200, deltaS<= 5e-5, maxF<= 1e-4, no constraints
TtSemiLepKinFitter::TtSemiLepKinFitter():
  TopKinFitter(),
  hadB_(0), hadP_(0), hadQ_(0), lepB_(0), lepton_(0), neutrino_(0),
  udscResolutions_(0), bResolutions_(0), lepResolutions_(0), metResolutions_(0),
  jetEnergyResolutionScaleFactors_(0), jetEnergyResolutionEtaBinning_(0),
  jetParam_(kEMom), lepParam_(kEMom), metParam_(kEMom)
{
  setupFitter();
}

TtSemiLepKinFitter::TtSemiLepKinFitter(Param jetParam, Param lepParam, Param metParam,
				       int maxNrIter, double maxDeltaS, double maxF,
				       const std::vector<Constraint>& constraints, double mW, double mTop,
				       const std::vector<edm::ParameterSet>* udscResolutions, 
				       const std::vector<edm::ParameterSet>* bResolutions,
				       const std::vector<edm::ParameterSet>* lepResolutions,
				       const std::vector<edm::ParameterSet>* metResolutions,
				       const std::vector<double>* jetEnergyResolutionScaleFactors,
				       const std::vector<double>* jetEnergyResolutionEtaBinning):
  TopKinFitter(maxNrIter, maxDeltaS, maxF, mW, mTop),
  hadB_(0), hadP_(0), hadQ_(0), lepB_(0), lepton_(0), neutrino_(0),
  udscResolutions_(udscResolutions), bResolutions_(bResolutions), lepResolutions_(lepResolutions), metResolutions_(metResolutions),
  jetEnergyResolutionScaleFactors_(jetEnergyResolutionScaleFactors), jetEnergyResolutionEtaBinning_(jetEnergyResolutionEtaBinning),
  jetParam_(jetParam), lepParam_(lepParam), metParam_(metParam), constrList_(constraints)
{
  setupFitter();
}

TtSemiLepKinFitter::~TtSemiLepKinFitter() 
{
  delete hadB_; 
  delete hadP_; 
  delete hadQ_;
  delete lepB_; 
  delete lepton_; 
  delete neutrino_;
  delete covM_;
  for(std::map<Constraint, TFitConstraintM*>::iterator it = massConstr_.begin(); it != massConstr_.end(); ++it)
    delete it->second;
  delete sumPxConstr_;
  delete sumPyConstr_;
}

void TtSemiLepKinFitter::printSetup() const
{
  std::stringstream constr;
  for(unsigned int i=0; i<constrList_.size(); ++i){
    switch(constrList_[i]){
    case kWHadMass       : constr << "    * hadronic W-mass (" << mW_   << " GeV) \n"; break;
    case kWLepMass       : constr << "    * leptonic W-mass (" << mW_   << " GeV) \n"; break;
    case kTopHadMass     : constr << "    * hadronic t-mass (" << mTop_ << " GeV) \n"; break;
    case kTopLepMass     : constr << "    * leptonic t-mass (" << mTop_ << " GeV) \n"; break;
    case kNeutrinoMass   : constr << "    * neutrino   mass (0 GeV) \n"; break;
    case kEqualTopMasses : constr << "    * equal    t-masses \n"; break;
    case kSumPt          : constr << "    * summed transverse momentum \n"; break;
    }
  }
  edm::LogVerbatim( "TtSemiLepKinFitter" ) 
    << "\n"
    << "+++++++++++ TtSemiLepKinFitter Setup ++++++++++++ \n"
    << "  Parametrization:                                \n" 
    << "   * jet : " << param(jetParam_) << "\n"
    << "   * lep : " << param(lepParam_) << "\n"
    << "   * met : " << param(metParam_) << "\n"
    << "  Constraints:                                    \n"
    <<    constr.str()
    << "  Max(No iterations): " << maxNrIter_ << "\n"
    << "  Max(deltaS)       : " << maxDeltaS_ << "\n"
    << "  Max(F)            : " << maxF_      << "\n"
    << "+++++++++++++++++++++++++++++++++++++++++++++++++ \n";
}

void TtSemiLepKinFitter::setupJets()
{
  TMatrixD empty3x3(3,3); 
  TMatrixD empty4x4(4,4);
  switch(jetParam_){ // setup jets according to parameterization
  case kEMom :
    hadB_= new TFitParticleEMomDev   ("Jet1", "Jet1", 0, &empty4x4);
    hadP_= new TFitParticleEMomDev   ("Jet2", "Jet2", 0, &empty4x4);
    hadQ_= new TFitParticleEMomDev   ("Jet3", "Jet3", 0, &empty4x4);
    lepB_= new TFitParticleEMomDev   ("Jet4", "Jet4", 0, &empty4x4);
    break;
  case kEtEtaPhi :
    hadB_= new TFitParticleEtEtaPhi  ("Jet1", "Jet1", 0, &empty3x3);
    hadP_= new TFitParticleEtEtaPhi  ("Jet2", "Jet2", 0, &empty3x3);
    hadQ_= new TFitParticleEtEtaPhi  ("Jet3", "Jet3", 0, &empty3x3);
    lepB_= new TFitParticleEtEtaPhi  ("Jet4", "Jet4", 0, &empty3x3);
    break;
  case kEtThetaPhi :
    hadB_= new TFitParticleEtThetaPhi("Jet1", "Jet1", 0, &empty3x3);
    hadP_= new TFitParticleEtThetaPhi("Jet2", "Jet2", 0, &empty3x3);
    hadQ_= new TFitParticleEtThetaPhi("Jet3", "Jet3", 0, &empty3x3);
    lepB_= new TFitParticleEtThetaPhi("Jet4", "Jet4", 0, &empty3x3);
    break;
  }
}

void TtSemiLepKinFitter::setupLeptons()
{
  TMatrixD empty3x3(3,3); 
  switch(lepParam_){ // setup lepton according to parameterization
  case kEMom       : lepton_  = new TFitParticleEScaledMomDev("Lepton",   "Lepton",   0, &empty3x3); break;
  case kEtEtaPhi   : lepton_  = new TFitParticleEtEtaPhi     ("Lepton",   "Lepton",   0, &empty3x3); break;
  case kEtThetaPhi : lepton_  = new TFitParticleEtThetaPhi   ("Lepton",   "Lepton",   0, &empty3x3); break;
  }
  switch(metParam_){ // setup neutrino according to parameterization
  case kEMom       : neutrino_= new TFitParticleEScaledMomDev("Neutrino", "Neutrino", 0, &empty3x3); break;
  case kEtEtaPhi   : neutrino_= new TFitParticleEtEtaPhi     ("Neutrino", "Neutrino", 0, &empty3x3); break;
  case kEtThetaPhi : neutrino_= new TFitParticleEtThetaPhi   ("Neutrino", "Neutrino", 0, &empty3x3); break;
  }
}

void TtSemiLepKinFitter::setupConstraints() 
{
  massConstr_[kWHadMass      ] = new TFitConstraintM("WMassHad",      "WMassHad",      0, 0, mW_  );
  massConstr_[kWLepMass      ] = new TFitConstraintM("WMassLep",      "WMassLep",      0, 0, mW_  );
  massConstr_[kTopHadMass    ] = new TFitConstraintM("TopMassHad",    "TopMassHad",    0, 0, mTop_);
  massConstr_[kTopLepMass    ] = new TFitConstraintM("TopMassLep",    "TopMassLep",    0, 0, mTop_);
  massConstr_[kNeutrinoMass  ] = new TFitConstraintM("NeutrinoMass",  "NeutrinoMass",  0, 0,    0.);
  massConstr_[kEqualTopMasses] = new TFitConstraintM("EqualTopMasses","EqualTopMasses",0, 0,    0.);
  sumPxConstr_                 = new TFitConstraintEp("SumPx",        "SumPx", 0, TFitConstraintEp::pX, 0.);
  sumPyConstr_                 = new TFitConstraintEp("SumPy",        "SumPy", 0, TFitConstraintEp::pY, 0.);

  massConstr_[kWHadMass      ]->addParticles1(hadP_,   hadQ_    );
  massConstr_[kWLepMass      ]->addParticles1(lepton_, neutrino_);
  massConstr_[kTopHadMass    ]->addParticles1(hadP_, hadQ_, hadB_);
  massConstr_[kTopLepMass    ]->addParticles1(lepton_, neutrino_, lepB_);
  massConstr_[kNeutrinoMass  ]->addParticle1 (neutrino_);
  massConstr_[kEqualTopMasses]->addParticles1(hadP_, hadQ_, hadB_);
  massConstr_[kEqualTopMasses]->addParticles2(lepton_, neutrino_, lepB_);
  sumPxConstr_->addParticles(lepton_, neutrino_, hadP_, hadQ_, hadB_, lepB_);
  sumPyConstr_->addParticles(lepton_, neutrino_, hadP_, hadQ_, hadB_, lepB_);

  if(std::find(constrList_.begin(), constrList_.end(), kSumPt)!=constrList_.end())
    constrainSumPt_ = true;
  constrainSumPt_ = false;
}

void TtSemiLepKinFitter::setupFitter() 
{
  printSetup();

  setupJets();
  setupLeptons();
  setupConstraints();

  // add measured particles
  fitter_->addMeasParticle(hadB_);
  fitter_->addMeasParticle(hadP_);
  fitter_->addMeasParticle(hadQ_);
  fitter_->addMeasParticle(lepB_);
  fitter_->addMeasParticle(lepton_);
  fitter_->addMeasParticle(neutrino_);

  // add constraints
  for(unsigned int i=0; i<constrList_.size(); i++){
    if(constrList_[i]!=kSumPt)
      fitter_->addConstraint(massConstr_[constrList_[i]]);
  }
  if(constrainSumPt_) {
    fitter_->addConstraint(sumPxConstr_);
    fitter_->addConstraint(sumPyConstr_);
  }

  // initialize helper class used to bring the resolutions into covariance matrices
  if(udscResolutions_->size() &&  bResolutions_->size() && lepResolutions_->size() && metResolutions_->size())
    covM_ = new CovarianceMatrix(*udscResolutions_, *bResolutions_, *lepResolutions_, *metResolutions_,
				 *jetEnergyResolutionScaleFactors_, *jetEnergyResolutionEtaBinning_);
  else
    covM_ = new CovarianceMatrix();
}

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

int TtSemiLepKinFitter::fit(const TLorentzVector& p4HadP, const TLorentzVector& p4HadQ, const TLorentzVector& p4HadB, const TLorentzVector& p4LepB,
			    const TLorentzVector& p4Lepton, const TLorentzVector& p4Neutrino, const int leptonCharge, const CovarianceMatrix::ObjectType leptonType)
{
  // initialize covariance matrices
  TMatrixD covHadP = covM_->setupMatrix(p4HadP, CovarianceMatrix::kUdscJet, jetParam_);
  TMatrixD covHadQ = covM_->setupMatrix(p4HadQ, CovarianceMatrix::kUdscJet, jetParam_);
  TMatrixD covHadB = covM_->setupMatrix(p4HadB, CovarianceMatrix::kBJet, jetParam_);
  TMatrixD covLepB = covM_->setupMatrix(p4LepB, CovarianceMatrix::kBJet, jetParam_);
  TMatrixD covLepton   = covM_->setupMatrix(p4Lepton  , leptonType             , lepParam_);
  TMatrixD covNeutrino = covM_->setupMatrix(p4Neutrino, CovarianceMatrix::kMet , metParam_);

  // now do the part that is fully independent of PAT features
  return fit(p4HadP, p4HadQ, p4HadB, p4LepB, p4Lepton, p4Neutrino,
	     covHadP, covHadQ, covHadB, covLepB, covLepton, covNeutrino,
	     leptonCharge);
}

int TtSemiLepKinFitter::fit(const TLorentzVector& p4HadP, const TLorentzVector& p4HadQ, const TLorentzVector& p4HadB, const TLorentzVector& p4LepB,
			    const TLorentzVector& p4Lepton, const TLorentzVector& p4Neutrino,
			    const TMatrixD& covHadP, const TMatrixD& covHadQ, const TMatrixD& covHadB, const TMatrixD& covLepB,
			    const TMatrixD& covLepton, const TMatrixD& covNeutrino, const int leptonCharge)
{
  // set the kinematics of the objects to be fitted
  hadP_->setIni4Vec( &p4HadP );
  hadQ_->setIni4Vec( &p4HadQ );
  hadB_->setIni4Vec( &p4HadB );
  lepB_->setIni4Vec( &p4LepB );
  lepton_->setIni4Vec( &p4Lepton );
  neutrino_->setIni4Vec( &p4Neutrino );

  hadP_->setCovMatrix( &covHadP );
  hadQ_->setCovMatrix( &covHadQ );
  hadB_->setCovMatrix( &covHadB );
  lepB_->setCovMatrix( &covLepB );
  lepton_  ->setCovMatrix( &covLepton   );
  neutrino_->setCovMatrix( &covNeutrino );

  if(constrainSumPt_){
    // setup Px and Py constraint for curent event configuration so that sum Pt will be conserved
    sumPxConstr_->setConstraint( p4HadP.Px() + p4HadQ.Px() + p4HadB.Px() + p4LepB.Px() + p4Lepton.Px() + p4Neutrino.Px() );
    sumPyConstr_->setConstraint( p4HadP.Py() + p4HadQ.Py() + p4HadB.Py() + p4LepB.Py() + p4Lepton.Py() + p4Neutrino.Py() );
  }

  // now do the fit
  fitter_->fit();

  // read back the resulting particles if the fit converged
  if(fitter_->getStatus()==0){
    // read back jet kinematics
    fittedHadP_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(hadP_->getCurr4Vec()->X(),
			       hadP_->getCurr4Vec()->Y(), hadP_->getCurr4Vec()->Z(), hadP_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedHadQ_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(hadQ_->getCurr4Vec()->X(),
			       hadQ_->getCurr4Vec()->Y(), hadQ_->getCurr4Vec()->Z(), hadQ_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedHadB_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(hadB_->getCurr4Vec()->X(),
			       hadB_->getCurr4Vec()->Y(), hadB_->getCurr4Vec()->Z(), hadB_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLepB_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lepB_->getCurr4Vec()->X(),
			       lepB_->getCurr4Vec()->Y(), lepB_->getCurr4Vec()->Z(), lepB_->getCurr4Vec()->E()), math::XYZPoint()));

    // read back lepton kinematics
    fittedLepton_= pat::Particle(reco::LeafCandidate(leptonCharge, math::XYZTLorentzVector(lepton_->getCurr4Vec()->X(),
				 lepton_->getCurr4Vec()->Y(), lepton_->getCurr4Vec()->Z(), lepton_->getCurr4Vec()->E()), math::XYZPoint()));

    // read back the MET kinematics
    fittedNeutrino_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(neutrino_->getCurr4Vec()->X(),
				   neutrino_->getCurr4Vec()->Y(), neutrino_->getCurr4Vec()->Z(), neutrino_->getCurr4Vec()->E()), math::XYZPoint()));

  }
  return fitter_->getStatus();
}

TtSemiEvtSolution TtSemiLepKinFitter::addKinFitInfo(TtSemiEvtSolution* asol) 
{

  TtSemiEvtSolution fitsol(*asol);

  std::vector<pat::Jet> jets;
  jets.resize(4);
  jets[TtSemiLepEvtPartons::LightQ   ] = fitsol.getCalHadp();
  jets[TtSemiLepEvtPartons::LightQBar] = fitsol.getCalHadq();
  jets[TtSemiLepEvtPartons::HadB     ] = fitsol.getCalHadb();
  jets[TtSemiLepEvtPartons::LepB     ] = fitsol.getCalLepb();

  // perform the fit, either using the electron or the muon
  if(fitsol.getDecay() == "electron") fit( jets, fitsol.getCalLepe(), fitsol.getCalLepn() );
  if(fitsol.getDecay() == "muon"    ) fit( jets, fitsol.getCalLepm(), fitsol.getCalLepn() );
  
  // add fitted information to the solution
  if (fitter_->getStatus() == 0) {
    // fill the fitted particles
    fitsol.setFitHadb( fittedHadB() );
    fitsol.setFitHadp( fittedHadP() );
    fitsol.setFitHadq( fittedHadQ() );
    fitsol.setFitLepb( fittedLepB() );
    fitsol.setFitLepl( fittedLepton() );
    fitsol.setFitLepn( fittedNeutrino() );
    // store the fit's chi2 probability
    fitsol.setProbChi2( fitProb() );
  }
  return fitsol;
}
