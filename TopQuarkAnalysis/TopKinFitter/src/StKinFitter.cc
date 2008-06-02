//
// $Id: StKinFitter.cc,v 1.3 2008/03/16 17:14:33 delaer Exp $
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

/* other parametrizations and constraints
#include "PhysicsTools/KinFitter/interface/TFitParticleESpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCPInvSpher.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"*/

StKinFitter::StKinFitter() :
    jetParam_(EMom), 
    lepParam_(EMom), 
    metParam_(EMom),
    maxNrIter_(200), 
    maxDeltaS_(5e-5), 
    maxF_(1e-4) 
{
  setupFitter();
}

StKinFitter::StKinFitter(int jetParam, int lepParam, int metParam,
			 int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints) :
  jetParam_((Parametrization) jetParam), 
  lepParam_((Parametrization) lepParam), 
  metParam_((Parametrization) metParam),
  maxNrIter_(maxNrIter), 
  maxDeltaS_(maxDeltaS), 
  maxF_(maxF),
  constraints_(constraints) 
{
  setupFitter();
}

StKinFitter::StKinFitter(Parametrization jetParam, Parametrization lepParam, Parametrization metParam,
                         int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints) :
  jetParam_(jetParam), 
  lepParam_(lepParam), 
  metParam_(metParam),
  maxNrIter_(maxNrIter), 
  maxDeltaS_(maxDeltaS), 
  maxF_(maxF),
  constraints_(constraints) 
{
  setupFitter();
}

StKinFitter::~StKinFitter() 
{
  delete cons1_; delete cons2_; delete cons3_;
  delete fitBottom_; delete fitLight_; delete fitLepton_; delete fitNeutrino_;
  delete theFitter_;
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
  if (jetParam_ == EMom) {
    m1b(0,0) = pow(fitsol.getBottom().resolutionA(), 2);
    m1b(1,1) = pow(fitsol.getBottom().resolutionB(), 2);
    m1b(2,2) = pow(fitsol.getBottom().resolutionC(), 2);
    m1b(3,3) = pow(fitsol.getBottom().resolutionD(), 2);
    m2b(0,0) = pow(fitsol.getLight ().resolutionA(), 2); 
    m2b(1,1) = pow(fitsol.getLight ().resolutionB(), 2); 
    m2b(2,2) = pow(fitsol.getLight ().resolutionC(), 2);
    m2b(3,3) = pow(fitsol.getLight ().resolutionD(), 2);
  } else if (jetParam_ == EtEtaPhi) {
    m1(0,0) = pow(fitsol.getBottom().resolutionEt (), 2);
    m1(1,1) = pow(fitsol.getBottom().resolutionEta(), 2);
    m1(2,2) = pow(fitsol.getBottom().resolutionPhi(), 2);
    m2(0,0) = pow(fitsol.getLight ().resolutionEt (), 2); 
    m2(1,1) = pow(fitsol.getLight ().resolutionEta(), 2); 
    m2(2,2) = pow(fitsol.getLight ().resolutionPhi(), 2);
  } else if (jetParam_ == EtThetaPhi) {
    m1(0,0) = pow(fitsol.getBottom().resolutionEt   (), 2);
    m1(1,1) = pow(fitsol.getBottom().resolutionTheta(), 2);
    m1(2,2) = pow(fitsol.getBottom().resolutionPhi  (), 2);
    m2(0,0) = pow(fitsol.getLight ().resolutionEt   (), 2); 
    m2(1,1) = pow(fitsol.getLight ().resolutionTheta(), 2); 
    m2(2,2) = pow(fitsol.getLight ().resolutionPhi  (), 2);
  }
  // lepton resolutions
  if (lepParam_ == EMom) {
    if(fitsol.getDecay()== "electron"){
      m3(0,0) = pow(fitsol.getElectron().resolutionA(), 2);
      m3(1,1) = pow(fitsol.getElectron().resolutionB(), 2); 
      m3(2,2) = pow(fitsol.getElectron().resolutionC(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m3(0,0) = pow(fitsol.getMuon().resolutionA(), 2);
      m3(1,1) = pow(fitsol.getMuon().resolutionB(), 2); 
      m3(2,2) = pow(fitsol.getMuon().resolutionC(), 2);
    }
  } else if (lepParam_ == EtEtaPhi) {
    if(fitsol.getDecay()== "electron"){
      m3(0,0) = pow(fitsol.getElectron().resolutionEt (), 2);
      m3(1,1) = pow(fitsol.getElectron().resolutionEta(), 2); 
      m3(2,2) = pow(fitsol.getElectron().resolutionPhi(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m3(0,0) = pow(fitsol.getMuon().resolutionEt (), 2);
      m3(1,1) = pow(fitsol.getMuon().resolutionEta(), 2); 
      m3(2,2) = pow(fitsol.getMuon().resolutionPhi(), 2);
    }
  } else if (lepParam_ == EtThetaPhi) {
    if(fitsol.getDecay()== "electron"){
      m3(0,0) = pow(fitsol.getElectron().resolutionEt   (), 2);
      m3(1,1) = pow(fitsol.getElectron().resolutionTheta(), 2); 
      m3(2,2) = pow(fitsol.getElectron().resolutionPhi  (), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m3(0,0) = pow(fitsol.getMuon().resolutionEt   (), 2);
      m3(1,1) = pow(fitsol.getMuon().resolutionTheta(), 2); 
      m3(2,2) = pow(fitsol.getMuon().resolutionPhi  (), 2);
    }
  }
  // neutrino resolutions
  if (metParam_ == EMom) {
    m4(0,0) = pow(fitsol.getNeutrino().resolutionA(), 2);
    m4(1,1) = pow(fitsol.getNeutrino().resolutionB(), 2);
    m4(2,2) = pow(fitsol.getNeutrino().resolutionC(), 2);
  } else if (metParam_ == EtEtaPhi) {
    m4(0,0) = pow(fitsol.getNeutrino().resolutionEt (), 2);
    m4(1,1) = pow(fitsol.getNeutrino().resolutionEta(), 2);
    m4(2,2) = pow(fitsol.getNeutrino().resolutionPhi(), 2);
  } else if (metParam_ == EtThetaPhi) {
    m4(0,0) = pow(fitsol.getNeutrino().resolutionEt   (), 2);
    m4(1,1) = pow(fitsol.getNeutrino().resolutionTheta(), 2);
    m4(2,2) = pow(fitsol.getNeutrino().resolutionPhi  (), 2);
  }

  // set the kinematics of the objects to be fitted
  fitBottom_->setIni4Vec(&bottomVec);
  fitLight_->setIni4Vec(&lightVec);
  fitLepton_->setIni4Vec(&leplVec);
  fitNeutrino_->setIni4Vec(&lepnVec);
  if (jetParam_ == EMom) {
    fitBottom_->setCovMatrix(&m1b);
    fitLight_->setCovMatrix(&m2b);
  } else {
    fitBottom_->setCovMatrix(&m1);
    fitLight_->setCovMatrix(&m2);
  }
  fitLepton_->setCovMatrix(&m3);
  fitNeutrino_->setCovMatrix(&m4);

  // perform the fit!
  theFitter_->fit();
  
  // add fitted information to the solution
  if (theFitter_->getStatus() == 0) {
    // read back the jet kinematics and resolutions
    pat::Particle aFitBottom(reco::LeafCandidate(0, math::XYZTLorentzVector(fitBottom_->getCurr4Vec()->X(), fitBottom_->getCurr4Vec()->Y(), fitBottom_->getCurr4Vec()->Z(), fitBottom_->getCurr4Vec()->E()),math::XYZPoint()));
    pat::Particle aFitLight(reco::LeafCandidate(0, math::XYZTLorentzVector(fitLight_->getCurr4Vec()->X(), fitLight_->getCurr4Vec()->Y(), fitLight_->getCurr4Vec()->Z(), fitLight_->getCurr4Vec()->E()),math::XYZPoint()));
    if (jetParam_ == EMom) {
      TMatrixD Vb(4,4); Vb = (*fitBottom_->getCovMatrixFit());
      aFitBottom.setCovMatrix(this->translateCovM(Vb));
      aFitBottom.setResolutionA(Vb(0,0));
      aFitBottom.setResolutionB(Vb(1,1));
      aFitBottom.setResolutionC(Vb(2,2));
      aFitBottom.setResolutionD(Vb(3,3));
      TMatrixD Vq(4,4); Vq = (*fitLight_->getCovMatrixFit());
      aFitLight.setCovMatrix(this->translateCovM(Vq));
      aFitLight.setResolutionA(Vq(0,0));
      aFitLight.setResolutionB(Vq(1,1));
      aFitLight.setResolutionC(Vq(2,2));
      aFitLight.setResolutionD(Vq(3,3));
    } else if (jetParam_ == EtEtaPhi) {
      TMatrixD Vb(3,3); Vb = (*fitBottom_->getCovMatrixFit());
      aFitBottom.setCovMatrix(this->translateCovM(Vb));
      aFitBottom.setResolutionEt(Vb(0,0));
      aFitBottom.setResolutionEta(Vb(1,1));
      aFitBottom.setResolutionPhi(Vb(2,2));
      TMatrixD Vq(3,3); Vq = (*fitLight_->getCovMatrixFit());
      aFitLight.setCovMatrix(this->translateCovM(Vq));
      aFitLight.setResolutionEt(Vq(0,0));
      aFitLight.setResolutionEta(Vq(1,1));
      aFitLight.setResolutionPhi(Vq(2,2));
    } else if (jetParam_ == EtThetaPhi) {
      TMatrixD Vb(3,3); Vb = (*fitBottom_->getCovMatrixFit());
      aFitBottom.setCovMatrix(this->translateCovM(Vb));
      aFitBottom.setResolutionEt(Vb(0,0));
      aFitBottom.setResolutionTheta(Vb(1,1));
      aFitBottom.setResolutionPhi(Vb(2,2));
      TMatrixD Vq(3,3); Vq = (*fitLight_->getCovMatrixFit());
      aFitLight.setCovMatrix(this->translateCovM(Vq));
      aFitLight.setResolutionEt(Vq(0,0));
      aFitLight.setResolutionTheta(Vq(1,1));
      aFitLight.setResolutionPhi(Vq(2,2));
    }
    // read back the lepton kinematics and resolutions
    pat::Particle aFitLepton(reco::LeafCandidate(0, math::XYZTLorentzVector(fitLepton_->getCurr4Vec()->X(), fitLepton_->getCurr4Vec()->Y(), fitLepton_->getCurr4Vec()->Z(), fitLepton_->getCurr4Vec()->E()), math::XYZPoint()));
    TMatrixD Vl(3,3); Vl = (*fitLepton_->getCovMatrixFit()); 
    aFitLepton.setCovMatrix(this->translateCovM(Vl));
    if (lepParam_ == EMom) {
      aFitLepton.setResolutionA(Vl(0,0));  
      aFitLepton.setResolutionB(Vl(1,1));
      aFitLepton.setResolutionC(Vl(2,2));
    } else if (lepParam_ == EtEtaPhi) {
      aFitLepton.setResolutionEt(Vl(0,0));  
      aFitLepton.setResolutionEta(Vl(1,1));
      aFitLepton.setResolutionPhi(Vl(2,2));
    } else if (lepParam_ == EtThetaPhi) {
      aFitLepton.setResolutionEt(Vl(0,0));  
      aFitLepton.setResolutionTheta(Vl(1,1));
      aFitLepton.setResolutionPhi(Vl(2,2));
    }
    // read back the MET kinematics and resolutions
    pat::Particle aFitNeutrino(reco::LeafCandidate(0, math::XYZTLorentzVector(fitNeutrino_->getCurr4Vec()->X(), fitNeutrino_->getCurr4Vec()->Y(), fitNeutrino_->getCurr4Vec()->Z(), fitNeutrino_->getCurr4Vec()->E()), math::XYZPoint()));   
    TMatrixD Vn(3,3); Vn = (*fitNeutrino_->getCovMatrixFit()); 
    aFitNeutrino.setCovMatrix(this->translateCovM(Vn));
    if (metParam_ == EMom) {
      aFitNeutrino.setResolutionA(Vn(0,0));  
      aFitNeutrino.setResolutionB(Vn(1,1));
      aFitNeutrino.setResolutionC(Vn(2,2));
    } else if (metParam_ == EtEtaPhi) {
      aFitNeutrino.setResolutionEt(Vn(0,0));  
      aFitNeutrino.setResolutionEta(Vn(1,1));
      aFitNeutrino.setResolutionPhi(Vn(2,2));
    } else if (metParam_ == EtThetaPhi) {
      aFitNeutrino.setResolutionEt(Vn(0,0));  
      aFitNeutrino.setResolutionTheta(Vn(1,1));
      aFitNeutrino.setResolutionPhi(Vn(2,2));
    }
    
    // finally fill the fitted particles
    fitsol.setFitBottom(aFitBottom);
    fitsol.setFitLight(aFitLight);
    fitsol.setFitLepton(aFitLepton);
    fitsol.setFitNeutrino(aFitNeutrino);

    // store the fit's chi2 probability
    fitsol.setChi2Prob(TMath::Prob(theFitter_->getS(), theFitter_->getNDF()));
  }

  return fitsol;

}

//
// Setup the fitter
//
void StKinFitter::setupFitter() {
  
  // FIXME: replace by messagelogger!!!
  
  cout<<endl<<endl<<"+++++++++++ KINFIT SETUP ++++++++++++"<<endl;
  cout<<"  jet parametrisation:     ";
  if(jetParam_ == EMom) cout<<"EMomDev"<<endl;
  if(jetParam_ == EtEtaPhi) cout<<"EtEtaPhi"<<endl;
  if(jetParam_ == EtThetaPhi) cout<<"EtThetaPhi"<<endl;
  cout<<"  lepton parametrisation:  ";
  if(lepParam_ == EMom) cout<<"EScaledMomDev"<<endl;
  if(lepParam_ == EtEtaPhi) cout<<"EtEtaPhi"<<endl;
  if(lepParam_ == EtThetaPhi) cout<<"EtThetaPhi"<<endl;
  cout<<"  met parametrisation:     ";
  if(metParam_ == EMom) cout<<"EScaledMomDev"<<endl;
  if(metParam_ == EtEtaPhi) cout<<"EtEtaPhi"<<endl;
  if(metParam_ == EtThetaPhi) cout<<"EtThetaPhi"<<endl;
  cout<<"  constraints:  "<<endl;
  for(unsigned int i=0; i<constraints_.size(); i++){
    if(constraints_[i] == 1) cout<<"    - hadronic W-mass"<<endl;
    if(constraints_[i] == 2) cout<<"    - leptonic W-mass"<<endl;
    if(constraints_[i] == 3) cout<<"    - hadronic top mass"<<endl;
    if(constraints_[i] == 4) cout<<"    - leptonic top mass"<<endl;
    if(constraints_[i] == 5) cout<<"    - neutrino mass"<<endl;
  }
  cout<<"Max. number of iterations: "<<maxNrIter_<<endl;
  cout<<"Max. deltaS: "<<maxDeltaS_<<endl;
  cout<<"Max. F: "<<maxF_<<endl;
  cout<<"++++++++++++++++++++++++++++++++++++++++++++"<<endl<<endl<<endl;
  
  theFitter_ = new TKinFitter("TtFit", "TtFit");

  TMatrixD empty3(3,3); TMatrixD empty4(4,4);
  if (jetParam_ == EMom) {
    fitBottom_ = new TFitParticleEMomDev("Jet1", "Jet1", 0, &empty4);
    fitLight_  = new TFitParticleEMomDev("Jet2", "Jet2", 0, &empty4);
  } else if (jetParam_ == EtEtaPhi) {
    fitBottom_ = new TFitParticleEtEtaPhi("Jet1", "Jet1", 0, &empty3);
    fitLight_  = new TFitParticleEtEtaPhi("Jet2", "Jet2", 0, &empty3);
  } else if (jetParam_ == EtThetaPhi) {
    fitBottom_ = new TFitParticleEtThetaPhi("Jet1", "Jet1", 0, &empty3);
    fitLight_  = new TFitParticleEtThetaPhi("Jet2", "Jet2", 0, &empty3);
  }
  if (lepParam_ == EMom) {
    fitLepton_ = new TFitParticleEScaledMomDev("Lepton", "Lepton", 0, &empty3);
  } else if (lepParam_ == EtEtaPhi) {
    fitLepton_ = new TFitParticleEtEtaPhi("Lepton", "Lepton", 0, &empty3);
  } else if (lepParam_ == EtThetaPhi) {
    fitLepton_ = new TFitParticleEtThetaPhi("Lepton", "Lepton", 0, &empty3);
  }
  if (metParam_ == EMom) {
    fitNeutrino_ = new TFitParticleEScaledMomDev("Neutrino", "Neutrino", 0, &empty3);
  } else if (metParam_ == EtEtaPhi) {
    fitNeutrino_ = new TFitParticleEtEtaPhi("Neutrino", "Neutrino", 0, &empty3);
  } else if (metParam_ == EtThetaPhi) {
    fitNeutrino_ = new TFitParticleEtThetaPhi("Neutrino", "Neutrino", 0, &empty3);
  }

  cons1_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0 , 80.35);
  cons1_->addParticles1(fitLepton_, fitNeutrino_);
  cons2_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 175.);
  cons2_->addParticles1(fitLepton_, fitNeutrino_, fitBottom_);
  cons3_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 0.);
  cons3_->addParticle1(fitNeutrino_);

  for (unsigned int i=0; i<constraints_.size(); i++) {
    if (constraints_[i] == 1) theFitter_->addConstraint(cons1_);
    if (constraints_[i] == 2) theFitter_->addConstraint(cons2_);
    if (constraints_[i] == 3) theFitter_->addConstraint(cons3_);
  }
  theFitter_->addMeasParticle(fitBottom_);
  theFitter_->addMeasParticle(fitLight_);
  theFitter_->addMeasParticle(fitLepton_);
  theFitter_->addMeasParticle(fitNeutrino_);

  theFitter_->setMaxNbIter(maxNrIter_);
  theFitter_->setMaxDeltaS(maxDeltaS_);
  theFitter_->setMaxF(maxF_);
  theFitter_->setVerbosity(0);
  
}

vector<float> StKinFitter::translateCovM(TMatrixD &V){
  vector<float> covM; 
  for(int ii=0; ii<V.GetNrows(); ii++){
    for(int jj=0; jj<V.GetNcols(); jj++) covM.push_back(V(ii,jj));
  }
  return covM;
}
