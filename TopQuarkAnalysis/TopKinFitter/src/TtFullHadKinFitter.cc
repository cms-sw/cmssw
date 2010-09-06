#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullHadKinFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// default constructor
TtFullHadKinFitter::TtFullHadKinFitter():
  TopKinFitter(),
  b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  jetParam_(kEMom)
{
  setupFitter();
  covM=0;
}

/// used to convert vector of int's to vector of constraints (just used in TtFullHadKinFitter(int, int, double, double, std::vector<unsigned int>))
std::vector<TtFullHadKinFitter::Constraint>
TtFullHadKinFitter::intToConstraint(std::vector<unsigned int> constraints)
{
  std::vector<TtFullHadKinFitter::Constraint> cConstraints;
  cConstraints.resize(constraints.size());
  for(unsigned int i=0;i<constraints.size();++i)
    {
      cConstraints[i]=(Constraint)constraints[i];
    }
  return cConstraints;
}

/// constructor initialized with build-in types as custom parameters (only included to keep TtHadEvtSolutionMaker.cc running)
TtFullHadKinFitter::TtFullHadKinFitter(int jetParam, int maxNrIter, double maxDeltaS, double maxF,
				       std::vector<unsigned int> constraints, double mW, double mTop):
  TopKinFitter(maxNrIter, maxDeltaS, maxF, mW, mTop),
  b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  jetParam_((Param)jetParam), constraints_(intToConstraint(constraints))
{
  setupFitter();
  covM=0;
}

/// constructor initialized with build-in types and class enum's custom parameters
TtFullHadKinFitter::TtFullHadKinFitter(Param jetParam, int maxNrIter, double maxDeltaS, double maxF,
				       std::vector<Constraint> constraints, double mW, double mTop):
  TopKinFitter(maxNrIter, maxDeltaS, maxF, mW, mTop),
  b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  jetParam_(jetParam), constraints_(constraints)
{
  setupFitter();
  covM=0;
}

/// default destructor
TtFullHadKinFitter::~TtFullHadKinFitter() 
{
  delete b_; 
  delete bBar_; 
  delete lightQ_;
  delete lightQBar_; 
  delete lightP_; 
  delete lightPBar_;
  delete covM;
  for(std::map<Constraint, TFitConstraintM*>::iterator it = massConstr_.begin(); it != massConstr_.end(); ++it)
    delete it->second;
}

/// print fitter setup
void 
TtFullHadKinFitter::printSetup() const
{
  std::stringstream constr;
  for(unsigned int i=0; i<constraints_.size(); ++i){
    switch(constraints_[i]){
    case kWPlusMass      : constr << "    * W+-mass   (" << mW_   << " GeV) \n"; break;
    case kWMinusMass     : constr << "    * W--mass   (" << mW_   << " GeV) \n"; break;
    case kTopMass        : constr << "    * t-mass    (" << mTop_ << " GeV) \n"; break;
    case kTopBarMass     : constr << "    * tBar-mass (" << mTop_ << " GeV) \n"; break;
    case kEqualTopMasses : constr << "    * equal t-masses \n"; break;
    }
  }
  edm::LogVerbatim( "TtFullHadKinFitter" ) 
    << "\n"
    << "+++++++++++ TtFullHadKinFitter Setup ++++++++++++ \n"
    << "  Parametrization:                                \n" 
    << "   * jet : " << param(jetParam_) << "\n"
    << "  Constraints:                                    \n"
    <<    constr.str()
    << "  Max(No iterations): " << maxNrIter_ << "\n"
    << "  Max(deltaS)       : " << maxDeltaS_ << "\n"
    << "  Max(F)            : " << maxF_      << "\n"
    << "+++++++++++++++++++++++++++++++++++++++++++++++++ \n";
}

/// initialize jet inputs
void 
TtFullHadKinFitter::setupJets()
{
  TMatrixD empty3x3(3,3); 
  TMatrixD empty4x4(4,4);
  switch(jetParam_){ // setup jets according to parameterization
  case kEMom :
    b_        = new TFitParticleEMomDev   ("Jet1", "Jet1", 0, &empty4x4);
    bBar_     = new TFitParticleEMomDev   ("Jet2", "Jet2", 0, &empty4x4);
    lightQ_   = new TFitParticleEMomDev   ("Jet3", "Jet3", 0, &empty4x4);
    lightQBar_= new TFitParticleEMomDev   ("Jet4", "Jet4", 0, &empty4x4);
    lightP_   = new TFitParticleEMomDev   ("Jet5", "Jet5", 0, &empty4x4);
    lightPBar_= new TFitParticleEMomDev   ("Jet6", "Jet6", 0, &empty4x4);
    break;
  case kEtEtaPhi :
    b_        = new TFitParticleEtEtaPhi  ("Jet1", "Jet1", 0, &empty3x3);
    bBar_     = new TFitParticleEtEtaPhi  ("Jet2", "Jet2", 0, &empty3x3);
    lightQ_   = new TFitParticleEtEtaPhi  ("Jet3", "Jet3", 0, &empty3x3);
    lightQBar_= new TFitParticleEtEtaPhi  ("Jet4", "Jet4", 0, &empty3x3);
    lightP_   = new TFitParticleEtEtaPhi  ("Jet5", "Jet5", 0, &empty3x3);
    lightPBar_= new TFitParticleEtEtaPhi  ("Jet6", "Jet6", 0, &empty3x3);
    break;
  case kEtThetaPhi :
    b_        = new TFitParticleEtThetaPhi("Jet1", "Jet1", 0, &empty3x3);
    bBar_     = new TFitParticleEtThetaPhi("Jet2", "Jet2", 0, &empty3x3);
    lightQ_   = new TFitParticleEtThetaPhi("Jet3", "Jet3", 0, &empty3x3);
    lightQBar_= new TFitParticleEtThetaPhi("Jet4", "Jet4", 0, &empty3x3);
    lightP_   = new TFitParticleEtThetaPhi("Jet5", "Jet5", 0, &empty3x3);
    lightPBar_= new TFitParticleEtThetaPhi("Jet6", "Jet6", 0, &empty3x3);
    break;
  }
}

/// initialize constraints
void 
TtFullHadKinFitter::setupConstraints() 
{
  massConstr_[kWPlusMass     ] = new TFitConstraintM("WPlusMass"     , "WPlusMass"      ,  0,  0, mW_   );
  massConstr_[kWMinusMass    ] = new TFitConstraintM("WMinusMass"    , "WMinusMass"     ,  0,  0, mW_   );
  massConstr_[kTopMass       ] = new TFitConstraintM("TopMass"       , "TopMass"        ,  0,  0, mTop_ );
  massConstr_[kTopBarMass    ] = new TFitConstraintM("TopBarMass"    , "TopBarMass"     ,  0,  0, mTop_ );
  massConstr_[kEqualTopMasses] = new TFitConstraintM("EqualTopMasses", "EqualTopMasses" ,  0,  0, 0     );
  
  massConstr_[kWPlusMass     ]->addParticles1(lightQ_, lightQBar_);
  massConstr_[kWMinusMass    ]->addParticles1(lightP_, lightPBar_);
  massConstr_[kTopMass       ]->addParticles1(b_, lightQ_, lightQBar_);
  massConstr_[kTopBarMass    ]->addParticles1(bBar_, lightP_, lightPBar_);
  massConstr_[kEqualTopMasses]->addParticles1(b_, lightQ_, lightQBar_);
  massConstr_[kEqualTopMasses]->addParticles2(bBar_, lightP_, lightPBar_);

}

/// setup fitter 
void 
TtFullHadKinFitter::setupFitter() 
{
  printSetup();
  setupJets();
  setupConstraints();

  // add measured particles
  fitter_->addMeasParticle(b_);
  fitter_->addMeasParticle(bBar_);
  fitter_->addMeasParticle(lightQ_);
  fitter_->addMeasParticle(lightQBar_);
  fitter_->addMeasParticle(lightP_);
  fitter_->addMeasParticle(lightPBar_);

  // add constraints
  for(unsigned int i=0; i<constraints_.size(); i++){
    fitter_->addConstraint(massConstr_[constraints_[i]]);
  }
}

/// kinematic fit interface
int 
TtFullHadKinFitter::fit(const std::vector<pat::Jet>& jets, const std::vector<edm::ParameterSet> udscResolutions, const std::vector<edm::ParameterSet> bResolutions)
{
  if( jets.size()<6 ){
    throw edm::Exception( edm::errors::Configuration, "Cannot run the TtFullHadKinFitter with less than 6 jets" );
  }

  // get jets in right order
  pat::Jet b         = jets[TtFullHadEvtPartons::B        ];
  pat::Jet bBar      = jets[TtFullHadEvtPartons::BBar     ];
  pat::Jet lightQ    = jets[TtFullHadEvtPartons::LightQ   ];
  pat::Jet lightQBar = jets[TtFullHadEvtPartons::LightQBar];
  pat::Jet lightP    = jets[TtFullHadEvtPartons::LightP   ];
  pat::Jet lightPBar = jets[TtFullHadEvtPartons::LightPBar];
 
  // initialize particles
  TLorentzVector p4B( b.px(), b.py(), b.pz(), b.energy() );
  TLorentzVector p4BBar( bBar.px(), bBar.py(), bBar.pz(), bBar.energy() );
  TLorentzVector p4LightQ( lightQ.px(), lightQ.py(), lightQ.pz(), lightQ.energy() );
  TLorentzVector p4LightQBar( lightQBar.px(), lightQBar.py(), lightQBar.pz(), lightQBar.energy() );
  TLorentzVector p4LightP( lightP.px(), lightP.py(), lightP.pz(), lightP.energy() );
  TLorentzVector p4LightPBar( lightPBar.px(), lightPBar.py(), lightPBar.pz(), lightPBar.energy() );

  // initialize covariance matrices
  if(!covM) covM = new CovarianceMatrix(udscResolutions, bResolutions);
  TMatrixD m1 = covM->setupMatrix(lightQ,    jetParam_);
  TMatrixD m2 = covM->setupMatrix(lightQBar, jetParam_);
  TMatrixD m3 = covM->setupMatrix(b,         jetParam_, "bjets");
  TMatrixD m4 = covM->setupMatrix(lightP,    jetParam_);
  TMatrixD m5 = covM->setupMatrix(lightPBar, jetParam_);
  TMatrixD m6 = covM->setupMatrix(bBar     , jetParam_, "bjets");

  // set the kinematics of the objects to be fitted
  b_        ->setIni4Vec(&p4B        );
  bBar_     ->setIni4Vec(&p4BBar     );
  lightQ_   ->setIni4Vec(&p4LightQ   );
  lightQBar_->setIni4Vec(&p4LightQBar);
  lightP_   ->setIni4Vec(&p4LightP   );
  lightPBar_->setIni4Vec(&p4LightPBar);
  
  // initialize covariance matrices
  lightQ_   ->setCovMatrix( &m1);
  lightQBar_->setCovMatrix( &m2);
  b_        ->setCovMatrix( &m3);
  lightP_   ->setCovMatrix( &m4);
  lightPBar_->setCovMatrix( &m5);
  bBar_     ->setCovMatrix( &m6);
  
  // perform the fit!
  fitter_->fit();
  
  // add fitted information to the solution
  if( fitter_->getStatus()==0 ){
    // read back jet kinematics
    fittedB_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(b_->getCurr4Vec()->X(), b_->getCurr4Vec()->Y(), b_->getCurr4Vec()->Z(), b_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightQ_   = pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightQ_->getCurr4Vec()->X(), lightQ_->getCurr4Vec()->Y(), lightQ_->getCurr4Vec()->Z(), lightQ_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightQBar_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightQBar_->getCurr4Vec()->X(), lightQBar_->getCurr4Vec()->Y(), lightQBar_->getCurr4Vec()->Z(), lightQBar_->getCurr4Vec()->E()), math::XYZPoint()));


    fittedBBar_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(bBar_->getCurr4Vec()->X(), bBar_->getCurr4Vec()->Y(), bBar_->getCurr4Vec()->Z(), bBar_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightP_   = pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightP_->getCurr4Vec()->X(), lightP_->getCurr4Vec()->Y(), lightP_->getCurr4Vec()->Z(), lightP_->getCurr4Vec()->E()), math::XYZPoint()));
    fittedLightPBar_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lightPBar_->getCurr4Vec()->X(), lightPBar_->getCurr4Vec()->Y(), lightPBar_->getCurr4Vec()->Z(), lightPBar_->getCurr4Vec()->E()), math::XYZPoint()));
  }
  return fitter_->getStatus();
}

/// kinematic fit interface
int 
TtFullHadKinFitter::fit(const std::vector<pat::Jet>& jets)
{
  const std::vector<edm::ParameterSet> emptyResolutionVector;
  return fit(jets, emptyResolutionVector, emptyResolutionVector);
}

/// add kin fit information to the old event solution (in for legacy reasons)
TtHadEvtSolution 
TtFullHadKinFitter::addKinFitInfo(TtHadEvtSolution * asol) 
{
  TtHadEvtSolution fitsol(*asol);

  std::vector<pat::Jet> jets;
  jets.resize(6);
  jets[TtFullHadEvtPartons::LightQ   ] = fitsol.getCalHadp();
  jets[TtFullHadEvtPartons::LightQBar] = fitsol.getCalHadq();
  jets[TtFullHadEvtPartons::B        ] = fitsol.getCalHadb();
  jets[TtFullHadEvtPartons::LightP   ] = fitsol.getCalHadj();
  jets[TtFullHadEvtPartons::LightPBar] = fitsol.getCalHadk();
  jets[TtFullHadEvtPartons::BBar     ] = fitsol.getCalHadbbar();

  fit( jets );

  // add fitted information to the solution
  if (fitter_->getStatus() == 0) {
    // finally fill the fitted particles
    fitsol.setFitHadb(fittedB_);
    fitsol.setFitHadp(fittedLightQ_);
    fitsol.setFitHadq(fittedLightQBar_);
    fitsol.setFitHadk(fittedLightP_);
    fitsol.setFitHadj(fittedLightPBar_);
    fitsol.setFitHadbbar(fittedBBar_);

    // store the fit's chi2 probability
    fitsol.setProbChi2( fitProb() );
  }
  return fitsol;
}
