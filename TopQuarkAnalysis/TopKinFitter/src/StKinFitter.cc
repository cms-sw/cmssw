//
// $Id: StKinFitter.cc,v 1.9 2013/05/30 20:51:27 gartung Exp $
//

#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"

#include "DataFormats/PatCandidates/interface/Particle.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitter.h"

//introduced to repair kinFit w/o resolutions from pat
#include "TopQuarkAnalysis/TopObjectResolutions/interface/MET.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Jet.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Muon.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/Electron.h"

/* other parametrizations and constraints
#include "PhysicsTools/KinFitter/interface/TFitParticleESpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCPInvSpher.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"*/

StKinFitter::StKinFitter() :
  TopKinFitter(),
  jetParam_(kEMom), 
  lepParam_(kEMom), 
  metParam_(kEMom)
{
  setupFitter();
}

StKinFitter::StKinFitter(int jetParam, int lepParam, int metParam,
			 int maxNrIter, double maxDeltaS, double maxF, const std::vector<int>& constraints) :
  TopKinFitter(maxNrIter, maxDeltaS, maxF),
  jetParam_((Param) jetParam), 
  lepParam_((Param) lepParam), 
  metParam_((Param) metParam),
  constraints_(constraints) 
{
  setupFitter();
}

StKinFitter::StKinFitter(Param jetParam, Param lepParam, Param metParam,
                         int maxNrIter, double maxDeltaS, double maxF, const std::vector<int>& constraints) :
  TopKinFitter(maxNrIter, maxDeltaS, maxF),
  jetParam_(jetParam),
  lepParam_(lepParam),
  metParam_(metParam),
  constraints_(constraints) 
{
  setupFitter();
}

StKinFitter::~StKinFitter() 
{
  delete cons1_; delete cons2_; delete cons3_;
  delete fitBottom_; delete fitLight_; delete fitLepton_; delete fitNeutrino_;
}

StEvtSolution StKinFitter::addKinFitInfo(StEvtSolution * asol) 
{
  StEvtSolution fitsol(*asol);

  TMatrixD m1(3,3),  m2(3,3);
  TMatrixD m1b(4,4), m2b(4,4);
  TMatrixD m3(3,3),  m4(3,3);
  m1.Zero();  m2.Zero();
  m1b.Zero(); m2b.Zero();
  m3.Zero();  m4.Zero();
  
  TLorentzVector bottomVec(fitsol.getBottom().px(),fitsol.getBottom().py(),
                           fitsol.getBottom().pz(),fitsol.getBottom().energy());
  TLorentzVector lightVec(fitsol.getLight().px(),fitsol.getLight().py(),
                      	  fitsol.getLight().pz(),fitsol.getLight().energy());
  TLorentzVector leplVec;
  if(fitsol.getDecay()== "electron") leplVec = TLorentzVector(fitsol.getElectron().px(), fitsol.getElectron().py(),    
			 				      fitsol.getElectron().pz(), fitsol.getElectron().energy());
  if(fitsol.getDecay()== "muon")     leplVec = TLorentzVector(fitsol.getMuon().px(), fitsol.getMuon().py(),    
			 				      fitsol.getMuon().pz(), fitsol.getMuon().energy());
  TLorentzVector lepnVec(fitsol.getNeutrino().px(), fitsol.getNeutrino().py(),
			 0, fitsol.getNeutrino().et());
    
  // jet resolutions
  {
    //FIXME this dirty hack needs a clean solution soon!
    double pt  = fitsol.getBottom().pt ();
    double eta = fitsol.getBottom().eta();
    res::HelperJet jetRes;
    if (jetParam_ == kEMom) {
      m1b(0,0) = pow(jetRes.pt (pt, eta, res::HelperJet::kB  ), 2);
      m1b(1,1) = pow(jetRes.pt (pt, eta, res::HelperJet::kB  ), 2);
      m1b(2,2) = pow(jetRes.pt (pt, eta, res::HelperJet::kB  ), 2);
      m1b(3,3) = pow(jetRes.pt (pt, eta, res::HelperJet::kB  ), 2);
      m2b(0,0) = pow(jetRes.pt (pt, eta, res::HelperJet::kUds), 2); 
      m2b(1,1) = pow(jetRes.pt (pt, eta, res::HelperJet::kUds), 2); 
      m2b(2,2) = pow(jetRes.pt (pt, eta, res::HelperJet::kUds), 2);
      m2b(3,3) = pow(jetRes.pt (pt, eta, res::HelperJet::kUds), 2);
    } else if (jetParam_ == kEtEtaPhi) {
      m1 (0,0) = pow(jetRes.pt (pt, eta, res::HelperJet::kB  ), 2);
      m1 (1,1) = pow(jetRes.eta(pt, eta, res::HelperJet::kB  ), 2);
      m1 (2,2) = pow(jetRes.phi(pt, eta, res::HelperJet::kB  ), 2);
      m2 (0,0) = pow(jetRes.pt (pt, eta, res::HelperJet::kUds), 2); 
      m2 (1,1) = pow(jetRes.eta(pt, eta, res::HelperJet::kUds), 2); 
      m2 (2,2) = pow(jetRes.phi(pt, eta, res::HelperJet::kUds), 2);
    } else if (jetParam_ == kEtThetaPhi) {
      m1 (0,0) = pow(jetRes.pt (pt, eta, res::HelperJet::kB  ), 2);
      m1 (1,1) = pow(jetRes.eta(pt, eta, res::HelperJet::kB  ), 2);
      m1 (2,2) = pow(jetRes.phi(pt, eta, res::HelperJet::kB  ), 2);
      m2 (0,0) = pow(jetRes.pt (pt, eta, res::HelperJet::kUds), 2); 
      m2 (1,1) = pow(jetRes.eta(pt, eta, res::HelperJet::kUds), 2); 
      m2 (2,2) = pow(jetRes.phi(pt, eta, res::HelperJet::kUds), 2);
    }
  }
  // lepton resolutions
  {
    //FIXME this dirty hack needs a clean solution soon!
    double pt  = fitsol.getElectron().pt ();
    double eta = fitsol.getElectron().eta();
    res::HelperMuon     muonRes;
    res::HelperElectron elecRes;
    if (lepParam_ == kEMom) {
      if(fitsol.getDecay()== "electron"){
	m3(0,0) = pow(elecRes.pt (pt, eta), 2);
	m3(1,1) = pow(elecRes.pt (pt, eta), 2); 
	m3(2,2) = pow(elecRes.pt (pt, eta), 2);
      }
      if(fitsol.getDecay()== "muon"){
	m3(0,0) = pow(muonRes.pt (pt, eta), 2);
	m3(1,1) = pow(muonRes.pt (pt, eta), 2); 
	m3(2,2) = pow(muonRes.pt (pt, eta), 2);
      }
    } else if (lepParam_ == kEtEtaPhi) {
      if(fitsol.getDecay()== "electron"){
	m3(0,0) = pow(elecRes.pt (pt, eta), 2);
	m3(1,1) = pow(elecRes.eta(pt, eta), 2); 
	m3(2,2) = pow(elecRes.phi(pt, eta), 2);
      }
      if(fitsol.getDecay()== "muon"){
	m3(0,0) = pow(muonRes.pt (pt, eta), 2);
	m3(1,1) = pow(muonRes.eta(pt, eta), 2); 
	m3(2,2) = pow(muonRes.phi(pt, eta), 2);
      }
    } else if (lepParam_ == kEtThetaPhi) {
      if(fitsol.getDecay()== "electron"){
	m3(0,0) = pow(elecRes.pt (pt, eta), 2);
	m3(1,1) = pow(elecRes.eta(pt, eta), 2); 
	m3(2,2) = pow(elecRes.phi(pt, eta), 2);
      }
      if(fitsol.getDecay()== "muon"){
	m3(0,0) = pow(muonRes.pt (pt, eta), 2);
	m3(1,1) = pow(muonRes.eta(pt, eta), 2); 
	m3(2,2) = pow(muonRes.phi(pt, eta), 2);
      }
    }
  }
  // neutrino resolutions
  {
    //FIXME this dirty hack needs a clean solution soon!
    double met = fitsol.getNeutrino().pt();
    res::HelperMET metRes;
    if (metParam_ == kEMom) {
      m4(0,0) = pow(metRes.met(met), 2);
      m4(1,1) = pow(         9999.,  2);
      m4(2,2) = pow(metRes.met(met), 2);
    } else if (metParam_ == kEtEtaPhi) {
      m4(0,0) = pow(metRes.met(met), 2);
      m4(1,1) = pow(         9999.,  2);
      m4(2,2) = pow(metRes.phi(met), 2);
    } else if (metParam_ == kEtThetaPhi) {
      m4(0,0) = pow(metRes.met(met), 2);
      m4(1,1) = pow(         9999.,  2);
      m4(2,2) = pow(metRes.phi(met), 2);
    }
  }
  // set the kinematics of the objects to be fitted
  fitBottom_->setIni4Vec(&bottomVec);
  fitLight_->setIni4Vec(&lightVec);
  fitLepton_->setIni4Vec(&leplVec);
  fitNeutrino_->setIni4Vec(&lepnVec);
  if (jetParam_ == kEMom) {
    fitBottom_->setCovMatrix(&m1b);
    fitLight_->setCovMatrix(&m2b);
  } else {
    fitBottom_->setCovMatrix(&m1);
    fitLight_->setCovMatrix(&m2);
  }
  fitLepton_->setCovMatrix(&m3);
  fitNeutrino_->setCovMatrix(&m4);

  // perform the fit!
  fitter_->fit();
  
  // add fitted information to the solution
  if (fitter_->getStatus() == 0) {
    // read back the jet kinematics and resolutions
    pat::Particle aFitBottom(reco::LeafCandidate(0, math::XYZTLorentzVector(fitBottom_->getCurr4Vec()->X(), fitBottom_->getCurr4Vec()->Y(), fitBottom_->getCurr4Vec()->Z(), fitBottom_->getCurr4Vec()->E()),math::XYZPoint()));
    pat::Particle aFitLight(reco::LeafCandidate(0, math::XYZTLorentzVector(fitLight_->getCurr4Vec()->X(), fitLight_->getCurr4Vec()->Y(), fitLight_->getCurr4Vec()->Z(), fitLight_->getCurr4Vec()->E()),math::XYZPoint()));

    // read back the lepton kinematics and resolutions
    pat::Particle aFitLepton(reco::LeafCandidate(0, math::XYZTLorentzVector(fitLepton_->getCurr4Vec()->X(), fitLepton_->getCurr4Vec()->Y(), fitLepton_->getCurr4Vec()->Z(), fitLepton_->getCurr4Vec()->E()), math::XYZPoint()));

    // read back the MET kinematics and resolutions
    pat::Particle aFitNeutrino(reco::LeafCandidate(0, math::XYZTLorentzVector(fitNeutrino_->getCurr4Vec()->X(), fitNeutrino_->getCurr4Vec()->Y(), fitNeutrino_->getCurr4Vec()->Z(), fitNeutrino_->getCurr4Vec()->E()), math::XYZPoint()));   
    
    // finally fill the fitted particles
    fitsol.setFitBottom(aFitBottom);
    fitsol.setFitLight(aFitLight);
    fitsol.setFitLepton(aFitLepton);
    fitsol.setFitNeutrino(aFitNeutrino);

    // store the fit's chi2 probability
    fitsol.setChi2Prob( fitProb() );
  }

  return fitsol;

}

//
// Setup the fitter
//
void StKinFitter::setupFitter() {
  
  // FIXME: replace by messagelogger!!!
  
  std::cout<<std::endl<<std::endl<<"+++++++++++ KINFIT SETUP ++++++++++++"<<std::endl;
  std::cout<<"  jet parametrisation:     " << param(jetParam_) << std::endl;
  std::cout<<"  lepton parametrisation:  " << param(lepParam_) << std::endl;
  std::cout<<"  met parametrisation:     " << param(metParam_) << std::endl;
  std::cout<<"  constraints:  "<<std::endl;
  for(unsigned int i=0; i<constraints_.size(); i++){
    if(constraints_[i] == 1) std::cout<<"    - hadronic W-mass"<<std::endl;
    if(constraints_[i] == 2) std::cout<<"    - leptonic W-mass"<<std::endl;
    if(constraints_[i] == 3) std::cout<<"    - hadronic top mass"<<std::endl;
    if(constraints_[i] == 4) std::cout<<"    - leptonic top mass"<<std::endl;
    if(constraints_[i] == 5) std::cout<<"    - neutrino mass"<<std::endl;
  }
  std::cout<<"Max. number of iterations: "<<maxNrIter_<<std::endl;
  std::cout<<"Max. deltaS: "<<maxDeltaS_<<std::endl;
  std::cout<<"Max. F: "<<maxF_<<std::endl;
  std::cout<<"++++++++++++++++++++++++++++++++++++++++++++"<<std::endl<<std::endl<<std::endl;

  TMatrixD empty3(3,3); TMatrixD empty4(4,4);
  if (jetParam_ == kEMom) {
    fitBottom_ = new TFitParticleEMomDev("Jet1", "Jet1", 0, &empty4);
    fitLight_  = new TFitParticleEMomDev("Jet2", "Jet2", 0, &empty4);
  } else if (jetParam_ == kEtEtaPhi) {
    fitBottom_ = new TFitParticleEtEtaPhi("Jet1", "Jet1", 0, &empty3);
    fitLight_  = new TFitParticleEtEtaPhi("Jet2", "Jet2", 0, &empty3);
  } else if (jetParam_ == kEtThetaPhi) {
    fitBottom_ = new TFitParticleEtThetaPhi("Jet1", "Jet1", 0, &empty3);
    fitLight_  = new TFitParticleEtThetaPhi("Jet2", "Jet2", 0, &empty3);
  }
  if (lepParam_ == kEMom) {
    fitLepton_ = new TFitParticleEScaledMomDev("Lepton", "Lepton", 0, &empty3);
  } else if (lepParam_ == kEtEtaPhi) {
    fitLepton_ = new TFitParticleEtEtaPhi("Lepton", "Lepton", 0, &empty3);
  } else if (lepParam_ == kEtThetaPhi) {
    fitLepton_ = new TFitParticleEtThetaPhi("Lepton", "Lepton", 0, &empty3);
  }
  if (metParam_ == kEMom) {
    fitNeutrino_ = new TFitParticleEScaledMomDev("Neutrino", "Neutrino", 0, &empty3);
  } else if (metParam_ == kEtEtaPhi) {
    fitNeutrino_ = new TFitParticleEtEtaPhi("Neutrino", "Neutrino", 0, &empty3);
  } else if (metParam_ == kEtThetaPhi) {
    fitNeutrino_ = new TFitParticleEtThetaPhi("Neutrino", "Neutrino", 0, &empty3);
  }

  cons1_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0 , mW_);
  cons1_->addParticles1(fitLepton_, fitNeutrino_);
  cons2_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, mTop_);
  cons2_->addParticles1(fitLepton_, fitNeutrino_, fitBottom_);
  cons3_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 0.);
  cons3_->addParticle1(fitNeutrino_);

  for (unsigned int i=0; i<constraints_.size(); i++) {
    if (constraints_[i] == 1) fitter_->addConstraint(cons1_);
    if (constraints_[i] == 2) fitter_->addConstraint(cons2_);
    if (constraints_[i] == 3) fitter_->addConstraint(cons3_);
  }
  fitter_->addMeasParticle(fitBottom_);
  fitter_->addMeasParticle(fitLight_);
  fitter_->addMeasParticle(fitLepton_);
  fitter_->addMeasParticle(fitNeutrino_);
  
}
