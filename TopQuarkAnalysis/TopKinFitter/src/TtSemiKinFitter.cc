//
// $Id$
//

#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiKinFitter.h"

#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtThetaPhi.h"
/*#include "PhysicsTools/KinFitter/interface/TFitParticleESpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCPInvSpher.h"
//#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"
*/

/// default constructor
TtSemiKinFitter::TtSemiKinFitter() :
    jetParam_(EMom), lepParam_(EMom), metParam_(EMom),
    maxNrIter_(200), maxDeltaS_(5e-5), maxF_(1e-4) {
  setupFitter();
}


/// constructor from configurables
TtSemiKinFitter::TtSemiKinFitter(Parametrization jetParam, Parametrization lepParam, Parametrization metParam,
                                 int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints) :
    jetParam_(jetParam), lepParam_(lepParam), metParam_(metParam),
    maxNrIter_(maxNrIter), maxDeltaS_(maxDeltaS), maxF_(maxF),
    constraints_(constraints) {
  setupFitter();
}


/// destructor
TtSemiKinFitter::~TtSemiKinFitter() {
  delete cons1_; delete cons2_; delete cons3_; delete cons4_; delete cons5_;
  delete fitHadb_; delete fitHadp_; delete fitHadq_;
  delete fitLepb_; delete fitLepl_; delete fitLepn_;
  delete theFitter_;
}


/// runs the fitter, and adds the altered kinematics to the event solution
TtSemiEvtSolution TtSemiKinFitter::addKinFitInfo(TtSemiEvtSolution * asol) {

  TtSemiEvtSolution fitsol(*asol);

  TMatrixD m1(3,3),  m2(3,3),  m3(3,3),  m4(3,3);
  TMatrixD m1b(4,4), m2b(4,4), m3b(4,4), m4b(4,4);
  TMatrixD m5(3,3), m6(3,3);
  m1.Zero();  m2.Zero();  m3.Zero();  m4.Zero();
  m1b.Zero(); m2b.Zero(); m3b.Zero(); m4b.Zero();
  m5.Zero();  m6.Zero();

  // initialize particles
  TLorentzVector hadpVec(fitsol.getCalHadp().px(), fitsol.getCalHadp().py(),
                         fitsol.getCalHadp().pz(), fitsol.getCalHadp().energy());
  TLorentzVector hadqVec(fitsol.getCalHadq().px(), fitsol.getCalHadq().py(),
                      	 fitsol.getCalHadq().pz(), fitsol.getCalHadq().energy());
  TLorentzVector hadbVec(fitsol.getCalHadb().px(), fitsol.getCalHadb().py(),
                         fitsol.getCalHadb().pz(), fitsol.getCalHadb().energy());
  TLorentzVector lepbVec(fitsol.getCalLepb().px(), fitsol.getCalLepb().py(),
                         fitsol.getCalLepb().pz(), fitsol.getCalLepb().energy());
  TLorentzVector leplVec;
  if(fitsol.getDecay()== "electron") leplVec = TLorentzVector(fitsol.getCalLepe().px(), fitsol.getCalLepe().py(),    
			 				      fitsol.getCalLepe().pz(), fitsol.getCalLepe().energy());
  if(fitsol.getDecay()== "muon")     leplVec = TLorentzVector(fitsol.getCalLepm().px(), fitsol.getCalLepm().py(),    
			 				      fitsol.getCalLepm().pz(), fitsol.getCalLepm().energy());
  TLorentzVector lepnVec(fitsol.getCalLepn().px(), fitsol.getCalLepn().py(),
			 0, fitsol.getCalLepn().et());

  // jet resolutions
  if (jetParam_ == EMom) {
    m1b(0,0) = pow(fitsol.getCalHadp().getResPinv(), 2);
    m1b(1,1) = pow(fitsol.getCalHadp().getResTheta(), 2);
    m1b(2,2) = pow(fitsol.getCalHadp().getResPhi(), 2);
    m1b(3,3) = pow(fitsol.getCalHadp().getResD(), 2);
    m2b(0,0) = pow(fitsol.getCalHadq().getResPinv(), 2); 
    m2b(1,1) = pow(fitsol.getCalHadq().getResTheta(), 2); 
    m2b(2,2) = pow(fitsol.getCalHadq().getResPhi(), 2);
    m2b(3,3) = pow(fitsol.getCalHadq().getResD(), 2);
    m3b(0,0) = pow(fitsol.getCalHadb().getResPinv(), 2); 
    m3b(1,1) = pow(fitsol.getCalHadb().getResTheta(), 2); 
    m3b(2,2) = pow(fitsol.getCalHadb().getResPhi(), 2);
    m3b(3,3) = pow(fitsol.getCalHadb().getResD(), 2);
    m4b(0,0) = pow(fitsol.getCalLepb().getResPinv(), 2); 
    m4b(1,1) = pow(fitsol.getCalLepb().getResTheta(), 2); 
    m4b(2,2) = pow(fitsol.getCalLepb().getResPhi(), 2);
    m4b(3,3) = pow(fitsol.getCalLepb().getResD(), 2);
  } else if (jetParam_ == EtEtaPhi) {
    m1(0,0) = pow(fitsol.getCalHadp().getResET(), 2);
    m1(1,1) = pow(fitsol.getCalHadp().getResEta(), 2);
    m1(2,2) = pow(fitsol.getCalHadp().getResPhi(), 2);
    m2(0,0) = pow(fitsol.getCalHadq().getResET(), 2); 
    m2(1,1) = pow(fitsol.getCalHadq().getResEta(), 2); 
    m2(2,2) = pow(fitsol.getCalHadq().getResPhi(), 2);
    m3(0,0) = pow(fitsol.getCalHadb().getResET(), 2); 
    m3(1,1) = pow(fitsol.getCalHadb().getResEta(), 2); 
    m3(2,2) = pow(fitsol.getCalHadb().getResPhi(), 2);
    m4(0,0) = pow(fitsol.getCalLepb().getResET(), 2); 
    m4(1,1) = pow(fitsol.getCalLepb().getResEta(), 2); 
    m4(2,2) = pow(fitsol.getCalLepb().getResPhi(), 2);
  } else if (jetParam_ == EtThetaPhi) {
    m1(0,0) = pow(fitsol.getCalHadp().getResET(), 2);
    m1(1,1) = pow(fitsol.getCalHadp().getResTheta(), 2);
    m1(2,2) = pow(fitsol.getCalHadp().getResPhi(), 2);
    m2(0,0) = pow(fitsol.getCalHadq().getResET(), 2); 
    m2(1,1) = pow(fitsol.getCalHadq().getResTheta(), 2); 
    m2(2,2) = pow(fitsol.getCalHadq().getResPhi(), 2);
    m3(0,0) = pow(fitsol.getCalHadb().getResET(), 2); 
    m3(1,1) = pow(fitsol.getCalHadb().getResTheta(), 2); 
    m3(2,2) = pow(fitsol.getCalHadb().getResPhi(), 2);
    m4(0,0) = pow(fitsol.getCalLepb().getResET(), 2); 
    m4(1,1) = pow(fitsol.getCalLepb().getResTheta(), 2); 
    m4(2,2) = pow(fitsol.getCalLepb().getResPhi(), 2);
  }
  // lepton resolutions
  if (lepParam_ == EMom) {
    if(fitsol.getDecay()== "electron"){
      m5(0,0) = pow(fitsol.getCalLepe().getResPinv(), 2);
      m5(1,1) = pow(fitsol.getCalLepe().getResTheta(), 2); 
      m5(2,2) = pow(fitsol.getCalLepe().getResPhi(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m5(0,0) = pow(fitsol.getCalLepm().getResPinv(), 2);
      m5(1,1) = pow(fitsol.getCalLepm().getResTheta(), 2); 
      m5(2,2) = pow(fitsol.getCalLepm().getResPhi(), 2);
    }
  } else if (jetParam_ == EtEtaPhi) {
    if(fitsol.getDecay()== "electron"){
      m5(0,0) = pow(fitsol.getRecLepe().getResET(), 2);
      m5(1,1) = pow(fitsol.getRecLepe().getResEta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepe().getResPhi(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m5(0,0) = pow(fitsol.getRecLepm().getResET(), 2);
      m5(1,1) = pow(fitsol.getRecLepm().getResEta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepm().getResPhi(), 2);
    }
  } else if (jetParam_ == EtThetaPhi) {
    if(fitsol.getDecay()== "electron") {
      m5(0,0) = pow(fitsol.getRecLepe().getResET(), 2);
      m5(1,1) = pow(fitsol.getRecLepe().getResTheta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepe().getResPhi(), 2);
    }
    if(fitsol.getDecay()== "muon") {
      m5(0,0) = pow(fitsol.getRecLepm().getResET(), 2);
      m5(1,1) = pow(fitsol.getRecLepm().getResTheta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepm().getResPhi(), 2);
    }
  }
  // neutrino resolutions
  if (lepParam_ == EMom) {
    m6(0,0) = pow(fitsol.getCalLepn().getResPinv(), 2);
    m6(1,1) = pow(fitsol.getCalLepn().getResTheta(), 2);
    m6(2,2) = pow(fitsol.getCalLepn().getResPhi(), 2);
  } else if (jetParam_ == EtEtaPhi) {
    m6(0,0) = pow(fitsol.getRecLepn().getResET(), 2);
    m6(1,1) = pow(fitsol.getRecLepn().getResEta(), 2);
    m6(2,2) = pow(fitsol.getRecLepn().getResPhi(), 2);
  } else if (jetParam_ == EtThetaPhi) {
    m6(0,0) = pow(fitsol.getRecLepn().getResET(), 2);
    m6(1,1) = pow(fitsol.getRecLepn().getResTheta(), 2);
    m6(2,2) = pow(fitsol.getRecLepn().getResPhi(), 2);
  }

  // set the kinematics of the objects to be fitted
  fitHadp_->setIni4Vec(&hadpVec);
  fitHadq_->setIni4Vec(&hadqVec);
  fitHadb_->setIni4Vec(&hadbVec);
  fitLepb_->setIni4Vec(&lepbVec);
  fitLepl_->setIni4Vec(&leplVec);
  fitLepn_->setIni4Vec(&lepnVec);
  if (jetParam_ == EMom) {
    fitHadp_->setCovMatrix(&m1b);
    fitHadq_->setCovMatrix(&m2b);
    fitHadb_->setCovMatrix(&m3b);
    fitLepb_->setCovMatrix(&m4b);
  } else {
    fitHadp_->setCovMatrix(&m1);
    fitHadq_->setCovMatrix(&m2);
    fitHadb_->setCovMatrix(&m3);
    fitLepb_->setCovMatrix(&m4);
  }
  fitLepl_->setCovMatrix(&m5);
  fitLepn_->setCovMatrix(&m6);

  // perform the fit!
  theFitter_->fit();
  
  // add fitted information to the solution
  if (theFitter_->getStatus() == 0) {
    // read back the jet kinematics and resolutions
    TopParticle aFitHadp(reco::Particle(0, math::XYZTLorentzVector(fitHadp_->getCurr4Vec()->X(), fitHadp_->getCurr4Vec()->Y(), fitHadp_->getCurr4Vec()->Z(), fitHadp_->getCurr4Vec()->E()), math::XYZPoint()));
    TopParticle aFitHadq(reco::Particle(0, math::XYZTLorentzVector(fitHadq_->getCurr4Vec()->X(), fitHadq_->getCurr4Vec()->Y(), fitHadq_->getCurr4Vec()->Z(), fitHadq_->getCurr4Vec()->E()), math::XYZPoint()));
    TopParticle aFitHadb(reco::Particle(0, math::XYZTLorentzVector(fitHadb_->getCurr4Vec()->X(), fitHadb_->getCurr4Vec()->Y(), fitHadb_->getCurr4Vec()->Z(), fitHadb_->getCurr4Vec()->E()), math::XYZPoint()));
    TopParticle aFitLepb(reco::Particle(0, math::XYZTLorentzVector(fitLepb_->getCurr4Vec()->X(), fitLepb_->getCurr4Vec()->Y(), fitLepb_->getCurr4Vec()->Z(), fitLepb_->getCurr4Vec()->E()), math::XYZPoint()));
    TMatrixD Vp(3,3);  Vp  = (*fitHadp_->getCovMatrixFit()); 
    TMatrixD Vq(3,3);  Vq  = (*fitHadq_->getCovMatrixFit()); 
    TMatrixD Vbh(3,3); Vbh = (*fitHadb_->getCovMatrixFit()); 
    TMatrixD Vbl(3,3); Vbl = (*fitLepb_->getCovMatrixFit());
    aFitHadp.setCovM(this->translateCovM(Vp));
    aFitHadq.setCovM(this->translateCovM(Vq));
    aFitHadb.setCovM(this->translateCovM(Vbh));
    aFitLepb.setCovM(this->translateCovM(Vbl));
    if (jetParam_ == EMom) {
      aFitHadp.setResPinv(sqrt(Vp(0,0)));  
      aFitHadp.setResTheta(sqrt(Vp(1,1)));
      aFitHadp.setResPhi(sqrt(Vp(2,2))); 
      aFitHadp.setResD(sqrt(Vp(3,3))); 
      aFitHadq.setResPinv(sqrt(Vq(0,0)));  
      aFitHadq.setResTheta(sqrt(Vq(1,1)));
      aFitHadq.setResPhi(sqrt(Vq(2,2)));
      aFitHadq.setResD(sqrt(Vq(3,3)));
      aFitHadb.setResPinv(sqrt(Vbh(0,0)));  
      aFitHadb.setResTheta(sqrt(Vbh(1,1)));
      aFitHadb.setResPhi(sqrt(Vbh(2,2)));
      aFitHadb.setResD(sqrt(Vbh(3,3)));
      aFitLepb.setResPinv(sqrt(Vbl(0,0)));  
      aFitLepb.setResTheta(sqrt(Vbl(1,1)));
      aFitLepb.setResPhi(sqrt(Vbl(2,2)));
      aFitLepb.setResD(sqrt(Vbl(3,3)));
    } else if (jetParam_ == EtEtaPhi) {
      aFitHadp.setResET (sqrt(Vp(0,0)));  
      aFitHadp.setResEta(sqrt(Vp(1,1)));
      aFitHadp.setResPhi(sqrt(Vp(2,2)));
      aFitHadq.setResET (sqrt(Vq(0,0)));  
      aFitHadq.setResEta(sqrt(Vq(1,1)));
      aFitHadq.setResPhi(sqrt(Vq(2,2)));
      aFitHadb.setResET (sqrt(Vbh(0,0)));  
      aFitHadb.setResEta(sqrt(Vbh(1,1)));
      aFitHadb.setResPhi(sqrt(Vbh(2,2)));
      aFitLepb.setResET (sqrt(Vbl(0,0)));  
      aFitLepb.setResEta(sqrt(Vbl(1,1)));
      aFitLepb.setResPhi(sqrt(Vbl(2,2)));
    } else if (jetParam_ == EtThetaPhi) {
      aFitHadp.setResET (sqrt(Vp(0,0)));  
      aFitHadp.setResTheta(sqrt(Vp(1,1)));
      aFitHadp.setResPhi(sqrt(Vp(2,2)));
      aFitHadq.setResET (sqrt(Vq(0,0)));  
      aFitHadq.setResTheta(sqrt(Vq(1,1)));
      aFitHadq.setResPhi(sqrt(Vq(2,2)));
      aFitHadb.setResET (sqrt(Vbh(0,0)));  
      aFitHadb.setResTheta(sqrt(Vbh(1,1)));
      aFitHadb.setResPhi(sqrt(Vbh(2,2)));
      aFitLepb.setResET (sqrt(Vbl(0,0)));  
      aFitLepb.setResTheta(sqrt(Vbl(1,1)));
      aFitLepb.setResPhi(sqrt(Vbl(2,2)));
    }
    // read back the lepton kinematics and resolutions
    TopParticle aFitLepl(reco::Particle(0, math::XYZTLorentzVector(fitLepl_->getCurr4Vec()->X(), fitLepl_->getCurr4Vec()->Y(), fitLepl_->getCurr4Vec()->Z(), fitLepl_->getCurr4Vec()->E()), math::XYZPoint()));
    TMatrixD Vl(3,3); Vl = (*fitLepl_->getCovMatrixFit()); 
    aFitLepl.setCovM(this->translateCovM(Vl));
    if (lepParam_ == EMom) {
      aFitLepl.setResPinv(Vl(0,0));
      aFitLepl.setResTheta(Vl(1,1));
      aFitLepl.setResPhi(Vl(2,2));
    } else if (lepParam_ == EtEtaPhi) {
      aFitLepl.setResET (sqrt(Vl(0,0)));  
      aFitLepl.setResTheta(sqrt(Vl(1,1)));
      aFitLepl.setResPhi(sqrt(Vl(2,2)));
    } else if (lepParam_ == EtThetaPhi) {
      aFitLepl.setResET (sqrt(Vl(0,0)));  
      aFitLepl.setResTheta(sqrt(Vl(1,1)));
      aFitLepl.setResPhi(sqrt(Vl(2,2)));
    }
    // read back the MET kinematics and resolutions
    TopParticle aFitLepn(reco::Particle(0, math::XYZTLorentzVector(fitLepn_->getCurr4Vec()->X(), fitLepn_->getCurr4Vec()->Y(), fitLepn_->getCurr4Vec()->Z(), fitLepn_->getCurr4Vec()->E()), math::XYZPoint()));   
    TMatrixD Vn(3,3); Vn = (*fitLepn_->getCovMatrixFit()); 
    aFitLepn.setCovM(this->translateCovM(Vn));
    if (metParam_ == EMom) {
      aFitLepn.setResPinv(Vn(0,0));
      aFitLepn.setResTheta(Vn(1,1));
      aFitLepn.setResPhi(Vn(2,2));
    } else if (metParam_ == EtEtaPhi) {
      aFitLepn.setResET (sqrt(Vn(0,0)));  
      aFitLepn.setResEta(sqrt(Vn(1,1)));
      aFitLepn.setResPhi(sqrt(Vn(2,2)));
    } else if (metParam_ == EtThetaPhi) {
      aFitLepn.setResET (sqrt(Vn(0,0)));  
      aFitLepn.setResTheta(sqrt(Vn(1,1)));
      aFitLepn.setResPhi(sqrt(Vn(2,2)));
    }

    // finally fill the fitted particles
    fitsol.setFitHadb(aFitHadb);
    fitsol.setFitHadp(aFitHadp);
    fitsol.setFitHadq(aFitHadq);
    fitsol.setFitLepb(aFitLepb);
    fitsol.setFitLepl(aFitLepl);
    fitsol.setFitLepn(aFitLepn);

    // store the fit's chi2 probability
    fitsol.setProbChi2(TMath::Prob(theFitter_->getS(), theFitter_->getNDF()));
  }

  return fitsol;

}


/// Method to setup the fitter
void TtSemiKinFitter::setupFitter() {
  
  theFitter_ = new TKinFitter("TtFit", "TtFit");

  TMatrixD empty3(3,3); TMatrixD empty4(4,4);
  if (jetParam_ == EMom) {
    fitHadb_ = new TFitParticleEMomDev("Jet1", "Jet1", 0, &empty4);
    fitHadp_ = new TFitParticleEMomDev("Jet2", "Jet2", 0, &empty4);
    fitHadq_ = new TFitParticleEMomDev("Jet3", "Jet3", 0, &empty4);
    fitLepb_ = new TFitParticleEMomDev("Jet4", "Jet4", 0, &empty4);
  } else if (jetParam_ == EtEtaPhi) {
    fitHadb_ = new TFitParticleEtEtaPhi("Jet1", "Jet1", 0, &empty3);
    fitHadp_ = new TFitParticleEtEtaPhi("Jet2", "Jet2", 0, &empty3);
    fitHadq_ = new TFitParticleEtEtaPhi("Jet3", "Jet3", 0, &empty3);
    fitLepb_ = new TFitParticleEtEtaPhi("Jet4", "Jet4", 0, &empty3);
  } else if (jetParam_ == EtThetaPhi) {
    fitHadb_ = new TFitParticleEtThetaPhi("Jet1", "Jet1", 0, &empty3);
    fitHadp_ = new TFitParticleEtThetaPhi("Jet2", "Jet2", 0, &empty3);
    fitHadq_ = new TFitParticleEtThetaPhi("Jet3", "Jet3", 0, &empty3);
    fitLepb_ = new TFitParticleEtThetaPhi("Jet4", "Jet4", 0, &empty3);
  }
  if (lepParam_ == EMom) {
    fitLepl_ = new TFitParticleEScaledMomDev("Lepton", "Lepton", 0, &empty3);
  } else if (lepParam_ == EtEtaPhi) {
    fitLepl_ = new TFitParticleEtEtaPhi("Lepton", "Lepton", 0, &empty3);
  } else if (lepParam_ == EtThetaPhi) {
    fitLepl_ = new TFitParticleEtThetaPhi("Lepton", "Lepton", 0, &empty3);
  }
  if (metParam_ == EMom) {
    fitLepn_ = new TFitParticleEScaledMomDev("Neutrino", "Neutrino", 0, &empty3);
  } else if (metParam_ == EtEtaPhi) {
    fitLepn_ = new TFitParticleEtEtaPhi("Neutrino", "Neutrino", 0, &empty3);
  } else if (metParam_ == EtThetaPhi) {
    fitLepn_ = new TFitParticleEtThetaPhi("Neutrino", "Neutrino", 0, &empty3);
  }

  cons1_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0 , 80.35);
  cons1_->addParticles1(fitHadp_, fitHadq_);
  cons2_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0 , 80.35);
  cons2_->addParticles1(fitLepl_, fitLepn_);
  cons3_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 175.);
  cons3_->addParticles1(fitHadp_, fitHadq_, fitHadb_);
  cons4_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 175.);
  cons4_->addParticles1(fitLepl_, fitLepn_, fitLepb_);
  cons5_ = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 0.);
  cons5_->addParticle1(fitLepn_);

  for(unsigned int i=0; i<constraints_.size(); i++){
    if(constraints_[i] == 1) theFitter_->addConstraint(cons1_);
    if(constraints_[i] == 2) theFitter_->addConstraint(cons2_);
    if(constraints_[i] == 3) theFitter_->addConstraint(cons3_);
    if(constraints_[i] == 4) theFitter_->addConstraint(cons4_);
    if(constraints_[i] == 5) theFitter_->addConstraint(cons5_);
  }
  theFitter_->addMeasParticle(fitHadb_);
  theFitter_->addMeasParticle(fitHadp_);
  theFitter_->addMeasParticle(fitHadq_);
  theFitter_->addMeasParticle(fitLepb_);
  theFitter_->addMeasParticle(fitLepl_);
  theFitter_->addMeasParticle(fitLepn_);

  theFitter_->setMaxNbIter(maxNrIter_);
  theFitter_->setMaxDeltaS(maxDeltaS_);
  theFitter_->setMaxF(maxF_);
  theFitter_->setVerbosity(0);

}


vector<double> TtSemiKinFitter::translateCovM(TMatrixD &V){
  vector<double> covM; 
  for(int ii=0; ii<V.GetNrows(); ii++){
    for(int jj=0; jj<V.GetNcols(); jj++) covM.push_back(V(ii,jj));
  }
  return covM;
}
