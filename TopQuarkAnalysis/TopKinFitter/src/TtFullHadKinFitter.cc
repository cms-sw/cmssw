#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullHadKinFitter.h"

//introduced to repair kinFit w/o resolutions from pat
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Jet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// default constructor
TtFullHadKinFitter::TtFullHadKinFitter():
  fitter_(0), b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  jetParam_(kEMom), maxNrIter_(200), maxDeltaS_(5e-5), maxF_(1e-4), mW_(80.4), mTop_(173.)
{
  setupFitter();
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
  fitter_(0), b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  jetParam_((Param)jetParam), maxNrIter_(maxNrIter), maxDeltaS_(maxDeltaS), maxF_(maxF),
  constraints_(intToConstraint(constraints)), mW_(mW), mTop_(mTop)
{
  setupFitter();
}

/// constructor initialized with build-in types and class enum's custom parameters
TtFullHadKinFitter::TtFullHadKinFitter(Param jetParam, int maxNrIter, double maxDeltaS, double maxF,
				       std::vector<Constraint> constraints, double mW, double mTop):
  fitter_(0), b_(0), bBar_(0), lightQ_(0), lightQBar_(0), lightP_(0), lightPBar_(0),
  jetParam_(jetParam), maxNrIter_(maxNrIter), maxDeltaS_(maxDeltaS), maxF_(maxF), constraints_(constraints), mW_(mW), mTop_(mTop)
{
  setupFitter();
}

/// default destructor
TtFullHadKinFitter::~TtFullHadKinFitter() 
{
  delete fitter_;
  delete b_; 
  delete bBar_; 
  delete lightQ_;
  delete lightQBar_; 
  delete lightP_; 
  delete lightPBar_;
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
    case kWPlusMass  : constr << "    * W+-mass   (" << mW_   << " GeV) \n"; break;
    case kWMinusMass : constr << "    * W--mass   (" << mW_   << " GeV) \n"; break;
    case kTopMass    : constr << "    * t-mass    (" << mTop_ << " GeV) \n"; break;
    case kTopBarMass : constr << "    * tBar-mass (" << mTop_ << " GeV) \n"; break;
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
  massConstr_[kWPlusMass ] = new TFitConstraintM("WPlusMass" , "WPlusMass"  ,  0,  0, mW_  );
  massConstr_[kWMinusMass] = new TFitConstraintM("WMinusMass", "WMinusMass" ,  0,  0, mW_  );
  massConstr_[kTopMass   ] = new TFitConstraintM("TopMass"   , "TopMass"    ,  0,  0, mTop_);
  massConstr_[kTopBarMass] = new TFitConstraintM("TopBarMass", "TopBarMass" ,  0,  0, mTop_);
  
  massConstr_[kWPlusMass ]->addParticles1(lightQ_, lightQBar_);
  massConstr_[kWMinusMass]->addParticles1(lightP_, lightPBar_);
  massConstr_[kTopMass   ]->addParticles1(b_, lightQ_, lightQBar_);
  massConstr_[kTopBarMass]->addParticles1(bBar_, lightP_, lightPBar_);
}

/// setup fitter 
void 
TtFullHadKinFitter::setupFitter() 
{
  printSetup();
  setupJets();
  setupConstraints();

  fitter_= new TKinFitter("TtFullHadronicFit", "TtFullHadronicFit");

  // configure fit
  fitter_->setMaxNbIter(maxNrIter_);
  fitter_->setMaxDeltaS(maxDeltaS_);
  fitter_->setMaxF(maxF_);
  fitter_->setVerbosity(0);

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
TtFullHadKinFitter::fit(const std::vector<pat::Jet>& jets)
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
  TMatrixD m1 (3,3), m2 (3,3), m3 (3,3), m4 (3,3);
  TMatrixD m1b(4,4), m2b(4,4), m3b(4,4), m4b(4,4);
  TMatrixD m5 (3,3), m5b(4,4), m6 (3,3), m6b(4,4);
  m1 .Zero(), m2 .Zero(), m3 .Zero(), m4 .Zero();
  m1b.Zero(), m2b.Zero(), m3b.Zero(), m4b.Zero();
  m5 .Zero(), m5b.Zero(), m6 .Zero(), m6b.Zero();

  // jet resolutions
  {
    //FIXME this dirty hack needs a clean solution soon!
    double q1pt  = p4LightQ.Pt() , q2pt  = p4LightQBar.Pt();
    double q3pt  = p4LightP.Pt() , q4pt  = p4LightPBar.Pt();
    double b1pt  = p4B     .Pt() , b2pt  = p4BBar     .Pt();
    double q1eta = p4LightQ.Eta(), q2eta = p4LightQBar.Eta();
    double q3eta = p4LightP.Eta(), q4eta = p4LightPBar.Eta();
    double b1eta = p4B     .Eta(), b2eta = p4BBar     .Eta();

    res::HelperJet jetRes;
    if( jetParam_==kEMom ){
      m1b(0,0) = pow(jetRes.a(q1pt, q1eta, res::HelperJet::kUds), 2);
      m1b(1,1) = pow(jetRes.b(q1pt, q1eta, res::HelperJet::kUds), 2);
      m1b(2,2) = pow(jetRes.c(q1pt, q1eta, res::HelperJet::kUds), 2);
      m1b(3,3) = pow(jetRes.d(q1pt, q1eta, res::HelperJet::kUds), 2);
      m2b(0,0) = pow(jetRes.a(q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2b(1,1) = pow(jetRes.b(q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2b(2,2) = pow(jetRes.c(q2pt, q2eta, res::HelperJet::kUds), 2);
      m2b(3,3) = pow(jetRes.d(q2pt, q2eta, res::HelperJet::kUds), 2);
      m3b(0,0) = pow(jetRes.a(b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3b(1,1) = pow(jetRes.b(b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3b(2,2) = pow(jetRes.c(b1pt, b1eta, res::HelperJet::kB  ), 2);
      m3b(3,3) = pow(jetRes.d(b1pt, b1eta, res::HelperJet::kB  ), 2);
      m4b(0,0) = pow(jetRes.a(q3pt, q3eta, res::HelperJet::kUds), 2);
      m4b(1,1) = pow(jetRes.b(q3pt, q3eta, res::HelperJet::kUds), 2);
      m4b(2,2) = pow(jetRes.c(q3pt, q3eta, res::HelperJet::kUds), 2);
      m4b(3,3) = pow(jetRes.d(q3pt, q3eta, res::HelperJet::kUds), 2);
      m5b(0,0) = pow(jetRes.a(q4pt, q4eta, res::HelperJet::kUds), 2); 
      m5b(1,1) = pow(jetRes.b(q4pt, q4eta, res::HelperJet::kUds), 2); 
      m5b(2,2) = pow(jetRes.c(q4pt, q4eta, res::HelperJet::kUds), 2);
      m5b(3,3) = pow(jetRes.d(q4pt, q4eta, res::HelperJet::kUds), 2);
      m6b(0,0) = pow(jetRes.a(b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m6b(1,1) = pow(jetRes.b(b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m6b(2,2) = pow(jetRes.c(b2pt, b2eta, res::HelperJet::kB  ), 2);
      m6b(3,3) = pow(jetRes.d(b2pt, b2eta, res::HelperJet::kB  ), 2);
    } else if( jetParam_==kEtEtaPhi ){
      m1 (0,0) = pow(jetRes.pt (q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (1,1) = pow(jetRes.eta(q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (2,2) = pow(jetRes.phi(q1pt, q1eta, res::HelperJet::kUds), 2);
      m2 (0,0) = pow(jetRes.pt (q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (1,1) = pow(jetRes.eta(q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (2,2) = pow(jetRes.phi(q2pt, q2eta, res::HelperJet::kUds), 2);
      m3 (0,0) = pow(jetRes.pt (b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (1,1) = pow(jetRes.eta(b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (2,2) = pow(jetRes.phi(b1pt, b1eta, res::HelperJet::kB  ), 2);
      m4 (0,0) = pow(jetRes.pt (q3pt, q3eta, res::HelperJet::kUds), 2);
      m4 (1,1) = pow(jetRes.eta(q3pt, q3eta, res::HelperJet::kUds), 2);
      m4 (2,2) = pow(jetRes.phi(q3pt, q3eta, res::HelperJet::kUds), 2);
      m5 (0,0) = pow(jetRes.pt (q4pt, q4eta, res::HelperJet::kUds), 2); 
      m5 (1,1) = pow(jetRes.eta(q4pt, q4eta, res::HelperJet::kUds), 2); 
      m5 (2,2) = pow(jetRes.phi(q4pt, q4eta, res::HelperJet::kUds), 2);
      m6 (0,0) = pow(jetRes.pt (b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m6 (1,1) = pow(jetRes.eta(b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m6 (2,2) = pow(jetRes.phi(b2pt, b2eta, res::HelperJet::kB  ), 2);
    } else if( jetParam_==kEtThetaPhi ) {
      m1 (0,0) = pow(jetRes.pt   (q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (1,1) = pow(jetRes.theta(q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (2,2) = pow(jetRes.phi  (q1pt, q1eta, res::HelperJet::kUds), 2);
      m2 (0,0) = pow(jetRes.pt   (q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (1,1) = pow(jetRes.theta(q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (2,2) = pow(jetRes.phi  (q2pt, q2eta, res::HelperJet::kUds), 2);
      m3 (0,0) = pow(jetRes.pt   (b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (1,1) = pow(jetRes.theta(b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (2,2) = pow(jetRes.phi  (b1pt, b1eta, res::HelperJet::kB  ), 2);
      m4 (0,0) = pow(jetRes.pt   (q3pt, q3eta, res::HelperJet::kUds), 2);
      m4 (1,1) = pow(jetRes.theta(q3pt, q3eta, res::HelperJet::kUds), 2);
      m4 (2,2) = pow(jetRes.phi  (q3pt, q3eta, res::HelperJet::kUds), 2);
      m5 (0,0) = pow(jetRes.pt   (q4pt, q4eta, res::HelperJet::kUds), 2); 
      m5 (1,1) = pow(jetRes.theta(q4pt, q4eta, res::HelperJet::kUds), 2); 
      m5 (2,2) = pow(jetRes.phi  (q4pt, q4eta, res::HelperJet::kUds), 2);
      m6 (0,0) = pow(jetRes.pt   (b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m6 (1,1) = pow(jetRes.theta(b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m6 (2,2) = pow(jetRes.phi  (b2pt, b2eta, res::HelperJet::kB  ), 2);
    }
  }

  // set the kinematics of the objects to be fitted
  b_        ->setIni4Vec(&p4B        );
  bBar_     ->setIni4Vec(&p4BBar     );
  lightQ_   ->setIni4Vec(&p4LightQ   );
  lightQBar_->setIni4Vec(&p4LightQBar);
  lightP_   ->setIni4Vec(&p4LightP   );
  lightPBar_->setIni4Vec(&p4LightPBar);
  
  if (jetParam_==kEMom) {
    lightQ_   ->setCovMatrix(&m1b);
    lightQBar_->setCovMatrix(&m2b);
    b_        ->setCovMatrix(&m3b);
    lightP_   ->setCovMatrix(&m4b);
    lightPBar_->setCovMatrix(&m5b);
    bBar_     ->setCovMatrix(&m6b);
  } else {
    lightQ_   ->setCovMatrix( &m1);
    lightQBar_->setCovMatrix( &m2);
    b_        ->setCovMatrix( &m3);
    lightP_   ->setCovMatrix( &m4);
    lightPBar_->setCovMatrix( &m5);
    bBar_     ->setCovMatrix( &m6);
  }
  
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
