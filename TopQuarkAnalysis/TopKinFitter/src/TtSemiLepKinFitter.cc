#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"

//introduced to repair kinFit w/o resolutions from pat
#include "TopQuarkAnalysis/TopObjectResolutions/interface/MET.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Jet.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Muon.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Electron.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//default configuration is: Parametrization kEMom,
//                          Max iterations =  200, 
//                          deltaS        <= 5e-5,
//                          maxF          <= 1e-4
//                          no constraints
TtSemiLepKinFitter::TtSemiLepKinFitter() : 
  fitter_(0), hadB_(0), hadP_(0), hadQ_(0), lepB_(0), lepton_(0), neutrino_(0),
  jetParam_(kEMom), lepParam_(kEMom), metParam_(kEMom), maxNrIter_(200), maxDeltaS_( 5e-5), maxF_(1e-4) 
{
  setupFitter();
}

TtSemiLepKinFitter::TtSemiLepKinFitter(Param  jetParam, 
				       Param  lepParam, 
				       Param  metParam,
				       int    maxNrIter, 
				       double maxDeltaS, 
				       double maxF, 
				       std::vector<Constraint> constr) :
  fitter_(0), hadB_(0), hadP_(0), hadQ_(0), lepB_(0), lepton_(0), neutrino_(0),
  jetParam_(jetParam), lepParam_(lepParam), metParam_(metParam), maxNrIter_(maxNrIter), 
  maxDeltaS_(maxDeltaS), maxF_(maxF), constrList_(constr) 
{
  setupFitter();
}

TtSemiLepKinFitter::~TtSemiLepKinFitter() 
{
  delete fitter_;
  delete hadB_; 
  delete hadP_; 
  delete hadQ_;
  delete lepB_; 
  delete lepton_; 
  delete neutrino_;
}

void TtSemiLepKinFitter::printSetup()
{
  std::string constr;
  for(unsigned int i=0; i<constrList_.size(); ++i){
    switch(constrList_[i]){
    case kWHadMass     : constr += "    * hadronic W-mass \n"; break;
    case kWLepMass     : constr += "    * leptonic W-mass \n"; break;
    case kTopHadMass   : constr += "    * hadronic t-mass \n"; break;
    case kTopLepMass   : constr += "    * leptonic t-mass \n"; break;
    case kNeutrinoMass : constr += "    * neutrino   mass \n"; break;
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
    <<    constr
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
  massConstr_[kWHadMass    ] = new TFitConstraintM("WMassHad",    "WMassHad",    0, 0 , 80.35);
  massConstr_[kWLepMass    ] = new TFitConstraintM("WMassLep",    "WMassLep",    0, 0 , 80.35);
  massConstr_[kTopHadMass  ] = new TFitConstraintM("TopMassHad",  "TopMassHad",  0, 0,   175.);
  massConstr_[kTopLepMass  ] = new TFitConstraintM("TopMassLep",  "TopMassLep",  0, 0,   175.);
  massConstr_[kNeutrinoMass] = new TFitConstraintM("NeutrinoMass","NeutrinoMass",0, 0,     0.);
  
  massConstr_[kWHadMass    ]->addParticles1(hadP_,   hadQ_    );
  massConstr_[kWLepMass    ]->addParticles1(lepton_, neutrino_);
  massConstr_[kTopHadMass  ]->addParticles1(hadP_, hadQ_, hadB_);
  massConstr_[kTopLepMass  ]->addParticles1(lepton_, neutrino_, lepB_);
  massConstr_[kNeutrinoMass]->addParticle1 (neutrino_);
}

void TtSemiLepKinFitter::setupFitter() 
{
  printSetup();

  setupJets();
  setupLeptons();
  setupConstraints();

  fitter_= new TKinFitter("TtSemiLeptonicFit", "TtSemiLeptonicFit");

  // configure fit
  fitter_->setMaxNbIter(maxNrIter_);
  fitter_->setMaxDeltaS(maxDeltaS_);
  fitter_->setMaxF(maxF_);
  fitter_->setVerbosity(0);

  // add measured particles
  fitter_->addMeasParticle(hadB_);
  fitter_->addMeasParticle(hadP_);
  fitter_->addMeasParticle(hadQ_);
  fitter_->addMeasParticle(lepB_);
  fitter_->addMeasParticle(lepton_);
  fitter_->addMeasParticle(neutrino_);

  // add constraints
  for(unsigned int i=0; i<constrList_.size(); i++){
    fitter_->addConstraint(massConstr_[constrList_[i]]);
  }
}

template <class LeptonType>
int TtSemiLepKinFitter::fit(const std::vector<pat::Jet>& jets, const pat::Lepton<LeptonType>& lepton, const pat::MET& neutrino)
{
  if( jets.size()<4 )
    throw edm::Exception( edm::errors::Configuration, "Cannot run the TtSemiLepKinFitter with less than 4 jets" );

  // get jets in right order
  pat::Jet hadP = jets[TtSemiLepEvtPartons::LightQ   ];
  pat::Jet hadQ = jets[TtSemiLepEvtPartons::LightQBar];
  pat::Jet hadB = jets[TtSemiLepEvtPartons::HadB     ];
  pat::Jet lepB = jets[TtSemiLepEvtPartons::LepB     ];
 
  // initialize particles
  TLorentzVector p4HadP( hadP.px(), hadP.py(), hadP.pz(), hadP.energy() );
  TLorentzVector p4HadQ( hadQ.px(), hadQ.py(), hadQ.pz(), hadQ.energy() );
  TLorentzVector p4HadB( hadB.px(), hadB.py(), hadB.pz(), hadB.energy() );
  TLorentzVector p4LepB( lepB.px(), lepB.py(), lepB.pz(), lepB.energy() );
  TLorentzVector p4Lepton  ( lepton.px(), lepton.py(), lepton.pz(), lepton.energy() );
  TLorentzVector p4Neutrino( neutrino.px(), neutrino.py(), 0, neutrino.et() );

  // initialize covariance matrices
  TMatrixD m1 (3,3), m2 (3,3), m3 (3,3), m4 (3,3);
  TMatrixD m1b(4,4), m2b(4,4), m3b(4,4), m4b(4,4);
  TMatrixD m5 (3,3), m6 (3,3);
  m1 .Zero(); m2 .Zero(); m3 .Zero(); m4 .Zero();
  m1b.Zero(); m2b.Zero(); m3b.Zero(); m4b.Zero();
  m5 .Zero(); m6 .Zero();

  // add jet resolutions
  {
    //FIXME this dirty hack needs a clean solution soon!
    double q1pt  = hadP.pt (), q2pt  = hadQ.pt ();
    double b1pt  = hadB.pt (), b2pt  = lepB.pt ();
    double q1eta = hadP.eta(), q2eta = hadQ.eta();
    double b1eta = hadB.eta(), b2eta = lepB.eta();
    
    res::HelperJet jetRes;
    switch(jetParam_){
    case kEMom :
      m1b(0,0) = pow(jetRes.a (q1pt, q1eta, res::HelperJet::kUds), 2);
      m1b(1,1) = pow(jetRes.b (q1pt, q1eta, res::HelperJet::kUds), 2);
      m1b(2,2) = pow(jetRes.c (q1pt, q1eta, res::HelperJet::kUds), 2);
      m1b(3,3) = pow(jetRes.d (q1pt, q1eta, res::HelperJet::kUds), 2);
      m2b(0,0) = pow(jetRes.a (q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2b(1,1) = pow(jetRes.b (q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2b(2,2) = pow(jetRes.c (q2pt, q2eta, res::HelperJet::kUds), 2);
      m2b(3,3) = pow(jetRes.d (q2pt, q2eta, res::HelperJet::kUds), 2);
      m3b(0,0) = pow(jetRes.a (b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3b(1,1) = pow(jetRes.b (b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3b(2,2) = pow(jetRes.c (b1pt, b1eta, res::HelperJet::kB  ), 2);
      m3b(3,3) = pow(jetRes.d (b1pt, b1eta, res::HelperJet::kB  ), 2);
      m4b(0,0) = pow(jetRes.a (b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m4b(1,1) = pow(jetRes.b (b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m4b(2,2) = pow(jetRes.c (b2pt, b2eta, res::HelperJet::kB  ), 2);
      m4b(3,3) = pow(jetRes.d (b2pt, b2eta, res::HelperJet::kB  ), 2);
      break;
    case kEtEtaPhi : 
      m1 (0,0) = pow(jetRes.et (q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (1,1) = pow(jetRes.eta(q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (2,2) = pow(jetRes.phi(q1pt, q1eta, res::HelperJet::kUds), 2);
      m2 (0,0) = pow(jetRes.et (q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (1,1) = pow(jetRes.eta(q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (2,2) = pow(jetRes.phi(q2pt, q2eta, res::HelperJet::kUds), 2);
      m3 (0,0) = pow(jetRes.et (b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (1,1) = pow(jetRes.eta(b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (2,2) = pow(jetRes.phi(b1pt, b1eta, res::HelperJet::kB  ), 2);
      m4 (0,0) = pow(jetRes.et (b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m4 (1,1) = pow(jetRes.eta(b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m4 (2,2) = pow(jetRes.phi(b2pt, b2eta, res::HelperJet::kB  ), 2);
      break;
    case kEtThetaPhi :
      m1 (0,0) = pow(jetRes.et   (q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (1,1) = pow(jetRes.theta(q1pt, q1eta, res::HelperJet::kUds), 2);
      m1 (2,2) = pow(jetRes.phi  (q1pt, q1eta, res::HelperJet::kUds), 2);
      m2 (0,0) = pow(jetRes.et   (q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (1,1) = pow(jetRes.theta(q2pt, q2eta, res::HelperJet::kUds), 2); 
      m2 (2,2) = pow(jetRes.phi  (q2pt, q2eta, res::HelperJet::kUds), 2);
      m3 (0,0) = pow(jetRes.et   (b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (1,1) = pow(jetRes.theta(b1pt, b1eta, res::HelperJet::kB  ), 2); 
      m3 (2,2) = pow(jetRes.phi  (b1pt, b1eta, res::HelperJet::kB  ), 2);
      m4 (0,0) = pow(jetRes.et   (b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m4 (1,1) = pow(jetRes.theta(b2pt, b2eta, res::HelperJet::kB  ), 2); 
      m4 (2,2) = pow(jetRes.phi  (b2pt, b2eta, res::HelperJet::kB  ), 2);
      break;
    }
  }

  // add lepton resolutions
  {
    //FIXME this dirty hack needs a clean solution soon!
    double pt  = lepton.pt ();
    double eta = lepton.eta();

    res::HelperMuon     muonRes;
    res::HelperElectron elecRes;
    switch(lepParam_){
    case kEMom :
      m5(0,0) = pow(elecRes.a (pt, eta), 2);
      m5(1,1) = pow(elecRes.b (pt, eta), 2); 
      m5(2,2) = pow(elecRes.c (pt, eta), 2);
      break;
    case kEtEtaPhi :
      m5(0,0) = pow(elecRes.et (pt, eta), 2);
      m5(1,1) = pow(elecRes.eta(pt, eta), 2); 
      m5(2,2) = pow(elecRes.phi(pt, eta), 2);
      break;
    case kEtThetaPhi :
      m5(0,0) = pow(elecRes.et (pt, eta), 2);
      m5(1,1) = pow(elecRes.theta(pt, eta), 2); 
      m5(2,2) = pow(elecRes.phi(pt, eta), 2);
      break;
    }
  }
  // add neutrino resolutions
  {
    //FIXME this dirty hack needs a clean solution soon!
    double pt = neutrino.pt();

    res::HelperMET metRes;
    switch(metParam_){
    case kEMom :
      m6(0,0) = pow(metRes.a(pt), 2);
      m6(1,1) = pow(metRes.b(pt), 2);
      m6(2,2) = pow(metRes.c(pt), 2);
      break;
    case kEtEtaPhi :
      m6(0,0) = pow(metRes.et(pt), 2);
      m6(1,1) = pow(          9999., 2);
      m6(2,2) = pow(metRes.phi(pt), 2);
      break;
    case kEtThetaPhi :
      m6(0,0) = pow(metRes.et(pt), 2);
      m6(1,1) = pow(          9999., 2);
      m6(2,2) = pow(metRes.phi(pt), 2);
      break;
    }
  }
  
  // set the kinematics of the objects to be fitted
  hadP_->setIni4Vec( &p4HadP );
  hadQ_->setIni4Vec( &p4HadQ );
  hadB_->setIni4Vec( &p4HadB );
  lepB_->setIni4Vec( &p4LepB );
  lepton_->setIni4Vec( &p4Lepton );
  neutrino_->setIni4Vec( &p4Neutrino );

  switch(jetParam_){
  case kEMom :
    hadP_->setCovMatrix( &m1b );
    hadQ_->setCovMatrix( &m2b );
    hadB_->setCovMatrix( &m3b );
    lepB_->setCovMatrix( &m4b );
    break;
  default :
    hadP_->setCovMatrix( &m1  );
    hadQ_->setCovMatrix( &m2  );
    hadB_->setCovMatrix( &m3  );
    lepB_->setCovMatrix( &m4  );
    break;
  }
  lepton_->setCovMatrix( &m5 );
  neutrino_->setCovMatrix( &m6 );

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

  // does not exist anymore in pat

//     // read back jet resolutions
//     switch(jetParam_){
//     case kEMom :
//       {
// 	TMatrixD Vp (4,4); Vp  = *hadP_->getCovMatrixFit(); 
// 	TMatrixD Vq (4,4); Vq  = *hadQ_->getCovMatrixFit(); 
// 	TMatrixD Vbh(4,4); Vbh = *hadB_->getCovMatrixFit(); 
// 	TMatrixD Vbl(4,4); Vbl = *lepB_->getCovMatrixFit();
// 	// covariance matices
// 	fittedHadP_.setCovMatrix( translateCovM(Vp ) );
// 	fittedHadQ_.setCovMatrix( translateCovM(Vq ) );
// 	fittedHadB_.setCovMatrix( translateCovM(Vbh) );
// 	fittedLepB_.setCovMatrix( translateCovM(Vbl) );
// 	// resolutions
// 	fittedHadP_.setResolutionA( sqrt(Vp (0,0)) );  
// 	fittedHadP_.setResolutionB( sqrt(Vp (1,1)) );
// 	fittedHadP_.setResolutionC( sqrt(Vp (2,2)) ); 
// 	fittedHadP_.setResolutionD( sqrt(Vp (3,3)) ); 
// 	fittedHadQ_.setResolutionA( sqrt(Vq (0,0)) );  
// 	fittedHadQ_.setResolutionB( sqrt(Vq (1,1)) );
// 	fittedHadQ_.setResolutionC( sqrt(Vq (2,2)) );
// 	fittedHadQ_.setResolutionD( sqrt(Vq (3,3)) );
// 	fittedHadB_.setResolutionA( sqrt(Vbh(0,0)) );  
// 	fittedHadB_.setResolutionB( sqrt(Vbh(1,1)) );
// 	fittedHadB_.setResolutionC( sqrt(Vbh(2,2)) );
// 	fittedHadB_.setResolutionD( sqrt(Vbh(3,3)) );
// 	fittedLepB_.setResolutionA( sqrt(Vbl(0,0)) );  
// 	fittedLepB_.setResolutionB( sqrt(Vbl(1,1)) );
// 	fittedLepB_.setResolutionC( sqrt(Vbl(2,2)) );
// 	fittedLepB_.setResolutionD( sqrt(Vbl(3,3)) );
// 	break;
//       }
//     case kEtEtaPhi :
//       {
// 	TMatrixD Vp (3,3); Vp  = *hadP_->getCovMatrixFit(); 
// 	TMatrixD Vq (3,3); Vq  = *hadQ_->getCovMatrixFit(); 
// 	TMatrixD Vbh(3,3); Vbh = *hadB_->getCovMatrixFit(); 
// 	TMatrixD Vbl(3,3); Vbl = *lepB_->getCovMatrixFit();
// 	// covariance matices
// 	fittedHadP_.setCovMatrix( translateCovM(Vp ) );
// 	fittedHadQ_.setCovMatrix( translateCovM(Vq ) );
// 	fittedHadB_.setCovMatrix( translateCovM(Vbh) );
// 	fittedLepB_.setCovMatrix( translateCovM(Vbl) );
// 	// resolutions
// 	fittedHadP_.setResolutionEt ( sqrt(Vp (0,0)) );  
// 	fittedHadP_.setResolutionEta( sqrt(Vp (1,1)) );
// 	fittedHadP_.setResolutionPhi( sqrt(Vp (2,2)) );
// 	fittedHadQ_.setResolutionEt ( sqrt(Vq (0,0)) );  
// 	fittedHadQ_.setResolutionEta( sqrt(Vq (1,1)) );
// 	fittedHadQ_.setResolutionPhi( sqrt(Vq (2,2)) );
// 	fittedHadB_.setResolutionEt ( sqrt(Vbh(0,0)) );  
// 	fittedHadB_.setResolutionEta( sqrt(Vbh(1,1)) );
// 	fittedHadB_.setResolutionPhi( sqrt(Vbh(2,2)) );
// 	fittedLepB_.setResolutionEt ( sqrt(Vbl(0,0)) );  
// 	fittedLepB_.setResolutionEta( sqrt(Vbl(1,1)) );
// 	fittedLepB_.setResolutionPhi( sqrt(Vbl(2,2)) );
// 	break;
//       }
//     case kEtThetaPhi :
//       {
// 	TMatrixD Vp (3,3); Vp  = *hadP_->getCovMatrixFit(); 
// 	TMatrixD Vq (3,3); Vq  = *hadQ_->getCovMatrixFit(); 
// 	TMatrixD Vbh(3,3); Vbh = *hadB_->getCovMatrixFit(); 
// 	TMatrixD Vbl(3,3); Vbl = *lepB_->getCovMatrixFit();
// 	// covariance matices
// 	fittedHadP_.setCovMatrix( translateCovM(Vp ) );
// 	fittedHadQ_.setCovMatrix( translateCovM(Vq ) );
// 	fittedHadB_.setCovMatrix( translateCovM(Vbh) );
// 	fittedLepB_.setCovMatrix( translateCovM(Vbl) );
// 	// resolutions
// 	fittedHadP_.setResolutionEt   ( sqrt(Vp (0,0)) );  
// 	fittedHadP_.setResolutionTheta( sqrt(Vp (1,1)) );
// 	fittedHadP_.setResolutionPhi  ( sqrt(Vp (2,2)) );
// 	fittedHadQ_.setResolutionEt   ( sqrt(Vq (0,0)) );  
// 	fittedHadQ_.setResolutionTheta( sqrt(Vq (1,1)) );
// 	fittedHadQ_.setResolutionPhi  ( sqrt(Vq (2,2)) );
// 	fittedHadB_.setResolutionEt   ( sqrt(Vbh(0,0)) );  
// 	fittedHadB_.setResolutionTheta( sqrt(Vbh(1,1)) );
// 	fittedHadB_.setResolutionPhi  ( sqrt(Vbh(2,2)) );
// 	fittedLepB_.setResolutionEt   ( sqrt(Vbl(0,0)) );  
// 	fittedLepB_.setResolutionTheta( sqrt(Vbl(1,1)) );
// 	fittedLepB_.setResolutionPhi  ( sqrt(Vbl(2,2)) );
// 	break;
//       }
//     }

    // read back lepton kinematics
    fittedLepton_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(lepton_->getCurr4Vec()->X(),
				 lepton_->getCurr4Vec()->Y(), lepton_->getCurr4Vec()->Z(), lepton_->getCurr4Vec()->E()), math::XYZPoint()));

    // does not exist anymore in pat

//     // read back lepton resolutions
//     TMatrixD Vl(3,3); Vl = *lepton_->getCovMatrixFit(); 
//     fittedLepton_.setCovMatrix( translateCovM(Vl) );
//     switch(lepParam_){
//     case kEMom :
//       fittedLepton_.setResolutionA( Vl(0,0) );
//       fittedLepton_.setResolutionB( Vl(1,1) );
//       fittedLepton_.setResolutionC( Vl(2,2) );
//       break;
//     case kEtEtaPhi :
//       fittedLepton_.setResolutionEt   ( sqrt(Vl(0,0)) );  
//       fittedLepton_.setResolutionTheta( sqrt(Vl(1,1)) );
//       fittedLepton_.setResolutionPhi  ( sqrt(Vl(2,2)) );
//       break;
//     case kEtThetaPhi :
//       fittedLepton_.setResolutionEt   ( sqrt(Vl(0,0)) );  
//       fittedLepton_.setResolutionTheta( sqrt(Vl(1,1)) );
//       fittedLepton_.setResolutionPhi  ( sqrt(Vl(2,2)) );
//       break;
//     }

    // read back the MET kinematics
    fittedNeutrino_= pat::Particle(reco::LeafCandidate(0, math::XYZTLorentzVector(neutrino_->getCurr4Vec()->X(),
				   neutrino_->getCurr4Vec()->Y(), neutrino_->getCurr4Vec()->Z(), neutrino_->getCurr4Vec()->E()), math::XYZPoint()));   


    // does not exist anymore in pat

//     // read back neutrino resolutions
//     TMatrixD Vn(3,3); Vn = *neutrino_->getCovMatrixFit(); 
//     fittedNeutrino_.setCovMatrix( translateCovM(Vn) );
//     switch(metParam_){
//     case kEMom :
//       fittedNeutrino_.setResolutionA( Vn(0,0) );
//       fittedNeutrino_.setResolutionB( Vn(1,1) );
//       fittedNeutrino_.setResolutionC( Vn(2,2) );
//       break;
//     case kEtEtaPhi :
//       fittedNeutrino_.setResolutionEt ( sqrt(Vn(0,0)) );  
//       fittedNeutrino_.setResolutionEta( sqrt(Vn(1,1)) );
//       fittedNeutrino_.setResolutionPhi( sqrt(Vn(2,2)) );
//       break;
//     case kEtThetaPhi :
//       fittedNeutrino_.setResolutionEt   ( sqrt(Vn(0,0)) );  
//       fittedNeutrino_.setResolutionTheta( sqrt(Vn(1,1)) );
//       fittedNeutrino_.setResolutionPhi  ( sqrt(Vn(2,2)) );
//       break;
//     }
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
  if(fitsol.getDecay() == "muon")     fit( jets, fitsol.getCalLepm(), fitsol.getCalLepn() );
  
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

vector<float> TtSemiLepKinFitter::translateCovM(TMatrixD &V){
  vector<float> covM; 
  for(int ii=0; ii<V.GetNrows(); ii++){
    for(int jj=0; jj<V.GetNcols(); jj++) covM.push_back(V(ii,jj));
  }
  return covM;
}
