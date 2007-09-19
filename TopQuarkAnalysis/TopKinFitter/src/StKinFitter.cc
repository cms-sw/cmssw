//
// $Id$
//

#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitter.h"

#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
/* other parametrizations and constraints
#include "PhysicsTools/KinFitter/interface/TFitParticleESpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCPInvSpher.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"*/


/// default constructor
StKinFitter::StKinFitter() :
    jetParam_(EMom), lepParam_(EMom), metParam_(EMom),
    maxNrIter_(200), maxDeltaS_(5e-5), maxF_(1e-4) {
  setupFitter();
}


/// constructor from configurables
StKinFitter::StKinFitter(int jetParam, int lepParam, int metParam,
                                 int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints) :
    jetParam_((Parametrization) jetParam), lepParam_((Parametrization) lepParam), metParam_((Parametrization) metParam),
    maxNrIter_(maxNrIter), maxDeltaS_(maxDeltaS), maxF_(maxF),
    constraints_(constraints) {
  setupFitter();
}


/// constructor from configurables
StKinFitter::StKinFitter(Parametrization jetParam, Parametrization lepParam, Parametrization metParam,
                         int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints) :
    jetParam_(jetParam), lepParam_(lepParam), metParam_(metParam),
    maxNrIter_(maxNrIter), maxDeltaS_(maxDeltaS), maxF_(maxF),
    constraints_(constraints) {
  setupFitter();
}


/// destructor
StKinFitter::~StKinFitter() {
  delete cons1_; delete cons2_; delete cons3_;
  delete fitBottom_; delete fitLight_; delete fitLepton_; delete fitNeutrino_;
  delete theFitter_;
}


StEvtSolution StKinFitter::addKinFitInfo(StEvtSolution * asol) {

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
    m1b(0,0) = pow(fitsol.getBottom().getResA(), 2);
    m1b(1,1) = pow(fitsol.getBottom().getResB(), 2);
    m1b(2,2) = pow(fitsol.getBottom().getResC(), 2);
    m1b(3,3) = pow(fitsol.getBottom().getResD(), 2);
    m2b(0,0) = pow(fitsol.getLight().getResA(), 2); 
    m2b(1,1) = pow(fitsol.getLight().getResB(), 2); 
    m2b(2,2) = pow(fitsol.getLight().getResC(), 2);
    m2b(3,3) = pow(fitsol.getLight().getResD(), 2);
  } else if (jetParam_ == EtEtaPhi) {
    m1(0,0) = pow(fitsol.getBottom().getResET(), 2);
    m1(1,1) = pow(fitsol.getBottom().getResEta(), 2);
    m1(2,2) = pow(fitsol.getBottom().getResPhi(), 2);
    m2(0,0) = pow(fitsol.getLight().getResET(), 2); 
    m2(1,1) = pow(fitsol.getLight().getResEta(), 2); 
    m2(2,2) = pow(fitsol.getLight().getResPhi(), 2);
  } else if (jetParam_ == EtThetaPhi) {
    m1(0,0) = pow(fitsol.getBottom().getResET(), 2);
    m1(1,1) = pow(fitsol.getBottom().getResTheta(), 2);
    m1(2,2) = pow(fitsol.getBottom().getResPhi(), 2);
    m2(0,0) = pow(fitsol.getLight().getResET(), 2); 
    m2(1,1) = pow(fitsol.getLight().getResTheta(), 2); 
    m2(2,2) = pow(fitsol.getLight().getResPhi(), 2);
  }
  // lepton resolutions
  if (lepParam_ == EMom) {
    if(fitsol.getDecay()== "electron"){
      m3(0,0) = pow(fitsol.getElectron().getResA(), 2);
      m3(1,1) = pow(fitsol.getElectron().getResB(), 2); 
      m3(2,2) = pow(fitsol.getElectron().getResC(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m3(0,0) = pow(fitsol.getMuon().getResA(), 2);
      m3(1,1) = pow(fitsol.getMuon().getResB(), 2); 
      m3(2,2) = pow(fitsol.getMuon().getResC(), 2);
    }
  } else if (lepParam_ == EtEtaPhi) {
    if(fitsol.getDecay()== "electron"){
      m3(0,0) = pow(fitsol.getElectron().getResET(), 2);
      m3(1,1) = pow(fitsol.getElectron().getResEta(), 2); 
      m3(2,2) = pow(fitsol.getElectron().getResPhi(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m3(0,0) = pow(fitsol.getMuon().getResET(), 2);
      m3(1,1) = pow(fitsol.getMuon().getResEta(), 2); 
      m3(2,2) = pow(fitsol.getMuon().getResPhi(), 2);
    }
  } else if (lepParam_ == EtThetaPhi) {
    if(fitsol.getDecay()== "electron"){
      m3(0,0) = pow(fitsol.getElectron().getResET(), 2);
      m3(1,1) = pow(fitsol.getElectron().getResTheta(), 2); 
      m3(2,2) = pow(fitsol.getElectron().getResPhi(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m3(0,0) = pow(fitsol.getMuon().getResET(), 2);
      m3(1,1) = pow(fitsol.getMuon().getResTheta(), 2); 
      m3(2,2) = pow(fitsol.getMuon().getResPhi(), 2);
    }
  }
  // neutrino resolutions
  if (metParam_ == EMom) {
    m4(0,0) = pow(fitsol.getNeutrino().getResA(), 2);
    m4(1,1) = pow(fitsol.getNeutrino().getResB(), 2);
    m4(2,2) = pow(fitsol.getNeutrino().getResC(), 2);
  } else if (metParam_ == EtEtaPhi) {
    m4(0,0) = pow(fitsol.getNeutrino().getResET(), 2);
    m4(1,1) = pow(fitsol.getNeutrino().getResEta(), 2);
    m4(2,2) = pow(fitsol.getNeutrino().getResPhi(), 2);
  } else if (metParam_ == EtThetaPhi) {
    m4(0,0) = pow(fitsol.getNeutrino().getResET(), 2);
    m4(1,1) = pow(fitsol.getNeutrino().getResTheta(), 2);
    m4(2,2) = pow(fitsol.getNeutrino().getResPhi(), 2);
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
    TopParticle aFitBottom(reco::Particle(0, math::XYZTLorentzVector(fitBottom_->getCurr4Vec()->X(), fitBottom_->getCurr4Vec()->Y(), fitBottom_->getCurr4Vec()->Z(), fitBottom_->getCurr4Vec()->E()),math::XYZPoint()));
    TopParticle aFitLight(reco::Particle(0, math::XYZTLorentzVector(fitLight_->getCurr4Vec()->X(), fitLight_->getCurr4Vec()->Y(), fitLight_->getCurr4Vec()->Z(), fitLight_->getCurr4Vec()->E()),math::XYZPoint()));
    if (jetParam_ == EMom) {
      TMatrixD Vb(4,4); Vb = (*fitBottom_->getCovMatrixFit());
      aFitBottom.setCovM(this->translateCovM(Vb));
      aFitBottom.setResA(Vb(0,0));
      aFitBottom.setResB(Vb(1,1));
      aFitBottom.setResC(Vb(2,2));
      aFitBottom.setResD(Vb(3,3));
      TMatrixD Vq(4,4); Vq = (*fitLight_->getCovMatrixFit());
      aFitLight.setCovM(this->translateCovM(Vq));
      aFitLight.setResA(Vq(0,0));
      aFitLight.setResB(Vq(1,1));
      aFitLight.setResC(Vq(2,2));
      aFitLight.setResD(Vq(3,3));
    } else if (jetParam_ == EtEtaPhi) {
      TMatrixD Vb(3,3); Vb = (*fitBottom_->getCovMatrixFit());
      aFitBottom.setCovM(this->translateCovM(Vb));
      aFitBottom.setResET(Vb(0,0));
      aFitBottom.setResEta(Vb(1,1));
      aFitBottom.setResPhi(Vb(2,2));
      TMatrixD Vq(3,3); Vq = (*fitLight_->getCovMatrixFit());
      aFitLight.setCovM(this->translateCovM(Vq));
      aFitLight.setResET(Vq(0,0));
      aFitLight.setResEta(Vq(1,1));
      aFitLight.setResPhi(Vq(2,2));
    } else if (jetParam_ == EtThetaPhi) {
      TMatrixD Vb(3,3); Vb = (*fitBottom_->getCovMatrixFit());
      aFitBottom.setCovM(this->translateCovM(Vb));
      aFitBottom.setResET(Vb(0,0));
      aFitBottom.setResTheta(Vb(1,1));
      aFitBottom.setResPhi(Vb(2,2));
      TMatrixD Vq(3,3); Vq = (*fitLight_->getCovMatrixFit());
      aFitLight.setCovM(this->translateCovM(Vq));
      aFitLight.setResET(Vq(0,0));
      aFitLight.setResTheta(Vq(1,1));
      aFitLight.setResPhi(Vq(2,2));
    }
    // read back the lepton kinematics and resolutions
    TopParticle aFitLepton(reco::Particle(0, math::XYZTLorentzVector(fitLepton_->getCurr4Vec()->X(), fitLepton_->getCurr4Vec()->Y(), fitLepton_->getCurr4Vec()->Z(), fitLepton_->getCurr4Vec()->E()), math::XYZPoint()));
    TMatrixD Vl(3,3); Vl = (*fitLepton_->getCovMatrixFit()); 
    aFitLepton.setCovM(this->translateCovM(Vl));
    if (lepParam_ == EMom) {
      aFitLepton.setResA(Vl(0,0));  
      aFitLepton.setResB(Vl(1,1));
      aFitLepton.setResC(Vl(2,2));
    } else if (lepParam_ == EtEtaPhi) {
      aFitLepton.setResET(Vl(0,0));  
      aFitLepton.setResEta(Vl(1,1));
      aFitLepton.setResPhi(Vl(2,2));
    } else if (lepParam_ == EtThetaPhi) {
      aFitLepton.setResET(Vl(0,0));  
      aFitLepton.setResTheta(Vl(1,1));
      aFitLepton.setResPhi(Vl(2,2));
    }
    // read back the MET kinematics and resolutions
    TopParticle aFitNeutrino(reco::Particle(0, math::XYZTLorentzVector(fitNeutrino_->getCurr4Vec()->X(), fitNeutrino_->getCurr4Vec()->Y(), fitNeutrino_->getCurr4Vec()->Z(), fitNeutrino_->getCurr4Vec()->E()), math::XYZPoint()));   
    TMatrixD Vn(3,3); Vn = (*fitNeutrino_->getCovMatrixFit()); 
    aFitNeutrino.setCovM(this->translateCovM(Vn));
    if (metParam_ == EMom) {
      aFitNeutrino.setResA(Vn(0,0));  
      aFitNeutrino.setResB(Vn(1,1));
      aFitNeutrino.setResC(Vn(2,2));
    } else if (metParam_ == EtEtaPhi) {
      aFitNeutrino.setResET(Vn(0,0));  
      aFitNeutrino.setResEta(Vn(1,1));
      aFitNeutrino.setResPhi(Vn(2,2));
    } else if (metParam_ == EtThetaPhi) {
      aFitNeutrino.setResET(Vn(0,0));  
      aFitNeutrino.setResTheta(Vn(1,1));
      aFitNeutrino.setResPhi(Vn(2,2));
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


vector<double> StKinFitter::translateCovM(TMatrixD &V){
  vector<double> covM; 
  for(int ii=0; ii<V.GetNrows(); ii++){
    for(int jj=0; jj<V.GetNcols(); jj++) covM.push_back(V(ii,jj));
  }
  return covM;
}
