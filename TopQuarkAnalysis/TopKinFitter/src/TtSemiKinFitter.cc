//
// $Id: TtSemiKinFitter.cc,v 1.7 2008/03/16 17:14:33 delaer Exp $
//

#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiKinFitter.h"

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

TtSemiKinFitter::TtSemiKinFitter() :
  jetParam_(EMom), 
  lepParam_(EMom), 
  metParam_(EMom),
  maxNrIter_(200), 
  maxDeltaS_(5e-5), 
  maxF_(1e-4) 
{
  setupFitter();
}

TtSemiKinFitter::TtSemiKinFitter(int jetParam, int lepParam, int metParam,
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

TtSemiKinFitter::TtSemiKinFitter(Parametrization jetParam, Parametrization lepParam, Parametrization metParam,
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

TtSemiKinFitter::~TtSemiKinFitter() 
{
  delete cons1_; delete cons2_; delete cons3_; delete cons4_; delete cons5_;
  delete fitHadb_; delete fitHadp_; delete fitHadq_;
  delete fitLepb_; delete fitLepl_; delete fitLepn_;
  delete theFitter_;
}

TtSemiEvtSolution TtSemiKinFitter::addKinFitInfo(TtSemiEvtSolution * asol) 
{
  TtSemiEvtSolution fitsol(*asol);

  TMatrixD m1(3,3),  m2(3,3),  m3(3,3),  m4(3,3);
  TMatrixD m1b(4,4), m2b(4,4), m3b(4,4), m4b(4,4);
  TMatrixD m5(3,3),  m6(3,3);
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
    m1b(0,0) = pow(fitsol.getCalHadp().resolutionA(), 2);
    m1b(1,1) = pow(fitsol.getCalHadp().resolutionB(), 2);
    m1b(2,2) = pow(fitsol.getCalHadp().resolutionC(), 2);
    m1b(3,3) = pow(fitsol.getCalHadp().resolutionD(), 2);
    m2b(0,0) = pow(fitsol.getCalHadq().resolutionA(), 2); 
    m2b(1,1) = pow(fitsol.getCalHadq().resolutionB(), 2); 
    m2b(2,2) = pow(fitsol.getCalHadq().resolutionC(), 2);
    m2b(3,3) = pow(fitsol.getCalHadq().resolutionD(), 2);
    m3b(0,0) = pow(fitsol.getCalHadb().resolutionA(), 2); 
    m3b(1,1) = pow(fitsol.getCalHadb().resolutionB(), 2); 
    m3b(2,2) = pow(fitsol.getCalHadb().resolutionC(), 2);
    m3b(3,3) = pow(fitsol.getCalHadb().resolutionD(), 2);
    m4b(0,0) = pow(fitsol.getCalLepb().resolutionA(), 2); 
    m4b(1,1) = pow(fitsol.getCalLepb().resolutionB(), 2); 
    m4b(2,2) = pow(fitsol.getCalLepb().resolutionC(), 2);
    m4b(3,3) = pow(fitsol.getCalLepb().resolutionD(), 2);
  } else if (jetParam_ == EtEtaPhi) {
    m1(0,0) = pow(fitsol.getCalHadp().resolutionEt(), 2);
    m1(1,1) = pow(fitsol.getCalHadp().resolutionEta(), 2);
    m1(2,2) = pow(fitsol.getCalHadp().resolutionPhi(), 2);
    m2(0,0) = pow(fitsol.getCalHadq().resolutionEt(), 2); 
    m2(1,1) = pow(fitsol.getCalHadq().resolutionEta(), 2); 
    m2(2,2) = pow(fitsol.getCalHadq().resolutionPhi(), 2);
    m3(0,0) = pow(fitsol.getCalHadb().resolutionEt(), 2); 
    m3(1,1) = pow(fitsol.getCalHadb().resolutionEta(), 2); 
    m3(2,2) = pow(fitsol.getCalHadb().resolutionPhi(), 2);
    m4(0,0) = pow(fitsol.getCalLepb().resolutionEt(), 2); 
    m4(1,1) = pow(fitsol.getCalLepb().resolutionEta(), 2); 
    m4(2,2) = pow(fitsol.getCalLepb().resolutionPhi(), 2);
  } else if (jetParam_ == EtThetaPhi) {
    m1(0,0) = pow(fitsol.getCalHadp().resolutionEt(), 2);
    m1(1,1) = pow(fitsol.getCalHadp().resolutionTheta(), 2);
    m1(2,2) = pow(fitsol.getCalHadp().resolutionPhi(), 2);
    m2(0,0) = pow(fitsol.getCalHadq().resolutionEt(), 2); 
    m2(1,1) = pow(fitsol.getCalHadq().resolutionTheta(), 2); 
    m2(2,2) = pow(fitsol.getCalHadq().resolutionPhi(), 2);
    m3(0,0) = pow(fitsol.getCalHadb().resolutionEt(), 2); 
    m3(1,1) = pow(fitsol.getCalHadb().resolutionTheta(), 2); 
    m3(2,2) = pow(fitsol.getCalHadb().resolutionPhi(), 2);
    m4(0,0) = pow(fitsol.getCalLepb().resolutionEt(), 2); 
    m4(1,1) = pow(fitsol.getCalLepb().resolutionTheta(), 2); 
    m4(2,2) = pow(fitsol.getCalLepb().resolutionPhi(), 2);
  }
  // lepton resolutions
  if (lepParam_ == EMom) {
    if(fitsol.getDecay()== "electron"){
      m5(0,0) = pow(fitsol.getCalLepe().resolutionA(), 2);
      m5(1,1) = pow(fitsol.getCalLepe().resolutionB(), 2); 
      m5(2,2) = pow(fitsol.getCalLepe().resolutionC(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m5(0,0) = pow(fitsol.getCalLepm().resolutionA(), 2);
      m5(1,1) = pow(fitsol.getCalLepm().resolutionB(), 2); 
      m5(2,2) = pow(fitsol.getCalLepm().resolutionC(), 2);
    }
  } else if (lepParam_ == EtEtaPhi) {
    if(fitsol.getDecay()== "electron"){
      m5(0,0) = pow(fitsol.getRecLepe().resolutionEt(), 2);
      m5(1,1) = pow(fitsol.getRecLepe().resolutionEta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepe().resolutionPhi(), 2);
    }
    if(fitsol.getDecay()== "muon"){
      m5(0,0) = pow(fitsol.getRecLepm().resolutionEt(), 2);
      m5(1,1) = pow(fitsol.getRecLepm().resolutionEta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepm().resolutionPhi(), 2);
    }
  } else if (lepParam_ == EtThetaPhi) {
    if(fitsol.getDecay()== "electron") {
      m5(0,0) = pow(fitsol.getRecLepe().resolutionEt(), 2);
      m5(1,1) = pow(fitsol.getRecLepe().resolutionTheta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepe().resolutionPhi(), 2);
    }
    if(fitsol.getDecay()== "muon") {
      m5(0,0) = pow(fitsol.getRecLepm().resolutionEt(), 2);
      m5(1,1) = pow(fitsol.getRecLepm().resolutionTheta(), 2); 
      m5(2,2) = pow(fitsol.getRecLepm().resolutionPhi(), 2);
    }
  }
  // neutrino resolutions
  if (metParam_ == EMom) {
    m6(0,0) = pow(fitsol.getCalLepn().resolutionA(), 2);
    m6(1,1) = pow(fitsol.getCalLepn().resolutionB(), 2);
    m6(2,2) = pow(fitsol.getCalLepn().resolutionC(), 2);
  } else if (metParam_ == EtEtaPhi) {
    m6(0,0) = pow(fitsol.getRecLepn().resolutionEt(), 2);
    m6(1,1) = pow(fitsol.getRecLepn().resolutionEta(), 2);
    m6(2,2) = pow(fitsol.getRecLepn().resolutionPhi(), 2);
  } else if (metParam_ == EtThetaPhi) {
    m6(0,0) = pow(fitsol.getRecLepn().resolutionEt(), 2);
    m6(1,1) = pow(fitsol.getRecLepn().resolutionTheta(), 2);
    m6(2,2) = pow(fitsol.getRecLepn().resolutionPhi(), 2);
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
    pat::Particle aFitHadp(reco::LeafCandidate(0, math::XYZTLorentzVector(fitHadp_->getCurr4Vec()->X(), fitHadp_->getCurr4Vec()->Y(), fitHadp_->getCurr4Vec()->Z(), fitHadp_->getCurr4Vec()->E()), math::XYZPoint()));
    pat::Particle aFitHadq(reco::LeafCandidate(0, math::XYZTLorentzVector(fitHadq_->getCurr4Vec()->X(), fitHadq_->getCurr4Vec()->Y(), fitHadq_->getCurr4Vec()->Z(), fitHadq_->getCurr4Vec()->E()), math::XYZPoint()));
    pat::Particle aFitHadb(reco::LeafCandidate(0, math::XYZTLorentzVector(fitHadb_->getCurr4Vec()->X(), fitHadb_->getCurr4Vec()->Y(), fitHadb_->getCurr4Vec()->Z(), fitHadb_->getCurr4Vec()->E()), math::XYZPoint()));
    pat::Particle aFitLepb(reco::LeafCandidate(0, math::XYZTLorentzVector(fitLepb_->getCurr4Vec()->X(), fitLepb_->getCurr4Vec()->Y(), fitLepb_->getCurr4Vec()->Z(), fitLepb_->getCurr4Vec()->E()), math::XYZPoint()));
    if (jetParam_ == EMom) {
      TMatrixD Vp(4,4);  Vp  = (*fitHadp_->getCovMatrixFit()); 
      TMatrixD Vq(4,4);  Vq  = (*fitHadq_->getCovMatrixFit()); 
      TMatrixD Vbh(4,4); Vbh = (*fitHadb_->getCovMatrixFit()); 
      TMatrixD Vbl(4,4); Vbl = (*fitLepb_->getCovMatrixFit());
      aFitHadp.setCovMatrix(this->translateCovM(Vp));
      aFitHadq.setCovMatrix(this->translateCovM(Vq));
      aFitHadb.setCovMatrix(this->translateCovM(Vbh));
      aFitLepb.setCovMatrix(this->translateCovM(Vbl));
      aFitHadp.setResolutionA(sqrt(Vp(0,0)));  
      aFitHadp.setResolutionB(sqrt(Vp(1,1)));
      aFitHadp.setResolutionC(sqrt(Vp(2,2))); 
      aFitHadp.setResolutionD(sqrt(Vp(3,3))); 
      aFitHadq.setResolutionA(sqrt(Vq(0,0)));  
      aFitHadq.setResolutionB(sqrt(Vq(1,1)));
      aFitHadq.setResolutionC(sqrt(Vq(2,2)));
      aFitHadq.setResolutionD(sqrt(Vq(3,3)));
      aFitHadb.setResolutionA(sqrt(Vbh(0,0)));  
      aFitHadb.setResolutionB(sqrt(Vbh(1,1)));
      aFitHadb.setResolutionC(sqrt(Vbh(2,2)));
      aFitHadb.setResolutionD(sqrt(Vbh(3,3)));
      aFitLepb.setResolutionA(sqrt(Vbl(0,0)));  
      aFitLepb.setResolutionB(sqrt(Vbl(1,1)));
      aFitLepb.setResolutionC(sqrt(Vbl(2,2)));
      aFitLepb.setResolutionD(sqrt(Vbl(3,3)));
    } else if (jetParam_ == EtEtaPhi) {
      TMatrixD Vp(3,3);  Vp  = (*fitHadp_->getCovMatrixFit()); 
      TMatrixD Vq(3,3);  Vq  = (*fitHadq_->getCovMatrixFit()); 
      TMatrixD Vbh(3,3); Vbh = (*fitHadb_->getCovMatrixFit()); 
      TMatrixD Vbl(3,3); Vbl = (*fitLepb_->getCovMatrixFit());
      aFitHadp.setCovMatrix(this->translateCovM(Vp));
      aFitHadq.setCovMatrix(this->translateCovM(Vq));
      aFitHadb.setCovMatrix(this->translateCovM(Vbh));
      aFitLepb.setCovMatrix(this->translateCovM(Vbl));
      aFitHadp.setResolutionEt (sqrt(Vp(0,0)));  
      aFitHadp.setResolutionEta(sqrt(Vp(1,1)));
      aFitHadp.setResolutionPhi(sqrt(Vp(2,2)));
      aFitHadq.setResolutionEt (sqrt(Vq(0,0)));  
      aFitHadq.setResolutionEta(sqrt(Vq(1,1)));
      aFitHadq.setResolutionPhi(sqrt(Vq(2,2)));
      aFitHadb.setResolutionEt (sqrt(Vbh(0,0)));  
      aFitHadb.setResolutionEta(sqrt(Vbh(1,1)));
      aFitHadb.setResolutionPhi(sqrt(Vbh(2,2)));
      aFitLepb.setResolutionEt (sqrt(Vbl(0,0)));  
      aFitLepb.setResolutionEta(sqrt(Vbl(1,1)));
      aFitLepb.setResolutionPhi(sqrt(Vbl(2,2)));
    } else if (jetParam_ == EtThetaPhi) {
      TMatrixD Vp(3,3);  Vp  = (*fitHadp_->getCovMatrixFit()); 
      TMatrixD Vq(3,3);  Vq  = (*fitHadq_->getCovMatrixFit()); 
      TMatrixD Vbh(3,3); Vbh = (*fitHadb_->getCovMatrixFit()); 
      TMatrixD Vbl(3,3); Vbl = (*fitLepb_->getCovMatrixFit());
      aFitHadp.setCovMatrix(this->translateCovM(Vp));
      aFitHadq.setCovMatrix(this->translateCovM(Vq));
      aFitHadb.setCovMatrix(this->translateCovM(Vbh));
      aFitLepb.setCovMatrix(this->translateCovM(Vbl));
      aFitHadp.setResolutionEt (sqrt(Vp(0,0)));  
      aFitHadp.setResolutionTheta(sqrt(Vp(1,1)));
      aFitHadp.setResolutionPhi(sqrt(Vp(2,2)));
      aFitHadq.setResolutionEt (sqrt(Vq(0,0)));  
      aFitHadq.setResolutionTheta(sqrt(Vq(1,1)));
      aFitHadq.setResolutionPhi(sqrt(Vq(2,2)));
      aFitHadb.setResolutionEt (sqrt(Vbh(0,0)));  
      aFitHadb.setResolutionTheta(sqrt(Vbh(1,1)));
      aFitHadb.setResolutionPhi(sqrt(Vbh(2,2)));
      aFitLepb.setResolutionEt (sqrt(Vbl(0,0)));  
      aFitLepb.setResolutionTheta(sqrt(Vbl(1,1)));
      aFitLepb.setResolutionPhi(sqrt(Vbl(2,2)));
    }
    // read back the lepton kinematics and resolutions
    pat::Particle aFitLepl(reco::LeafCandidate(0, math::XYZTLorentzVector(fitLepl_->getCurr4Vec()->X(), fitLepl_->getCurr4Vec()->Y(), fitLepl_->getCurr4Vec()->Z(), fitLepl_->getCurr4Vec()->E()), math::XYZPoint()));
    TMatrixD Vl(3,3); Vl = (*fitLepl_->getCovMatrixFit()); 
    aFitLepl.setCovMatrix(this->translateCovM(Vl));
    if (lepParam_ == EMom) {
      aFitLepl.setResolutionA(Vl(0,0));
      aFitLepl.setResolutionB(Vl(1,1));
      aFitLepl.setResolutionC(Vl(2,2));
    } else if (lepParam_ == EtEtaPhi) {
      aFitLepl.setResolutionEt (sqrt(Vl(0,0)));  
      aFitLepl.setResolutionTheta(sqrt(Vl(1,1)));
      aFitLepl.setResolutionPhi(sqrt(Vl(2,2)));
    } else if (lepParam_ == EtThetaPhi) {
      aFitLepl.setResolutionEt (sqrt(Vl(0,0)));  
      aFitLepl.setResolutionTheta(sqrt(Vl(1,1)));
      aFitLepl.setResolutionPhi(sqrt(Vl(2,2)));
    }
    // read back the MEt kinematics and resolutions
    pat::Particle aFitLepn(reco::LeafCandidate(0, math::XYZTLorentzVector(fitLepn_->getCurr4Vec()->X(), fitLepn_->getCurr4Vec()->Y(), fitLepn_->getCurr4Vec()->Z(), fitLepn_->getCurr4Vec()->E()), math::XYZPoint()));   
    TMatrixD Vn(3,3); Vn = (*fitLepn_->getCovMatrixFit()); 
    aFitLepn.setCovMatrix(this->translateCovM(Vn));
    if (metParam_ == EMom) {
      aFitLepn.setResolutionA(Vn(0,0));
      aFitLepn.setResolutionB(Vn(1,1));
      aFitLepn.setResolutionC(Vn(2,2));
    } else if (metParam_ == EtEtaPhi) {
      aFitLepn.setResolutionEt (sqrt(Vn(0,0)));  
      aFitLepn.setResolutionEta(sqrt(Vn(1,1)));
      aFitLepn.setResolutionPhi(sqrt(Vn(2,2)));
    } else if (metParam_ == EtThetaPhi) {
      aFitLepn.setResolutionEt (sqrt(Vn(0,0)));  
      aFitLepn.setResolutionTheta(sqrt(Vn(1,1)));
      aFitLepn.setResolutionPhi(sqrt(Vn(2,2)));
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

void TtSemiKinFitter::setupFitter() 
{  
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

vector<float> TtSemiKinFitter::translateCovM(TMatrixD &V){
  vector<float> covM; 
  for(int ii=0; ii<V.GetNrows(); ii++){
    for(int jj=0; jj<V.GetNcols(); jj++) covM.push_back(V(ii,jj));
  }
  return covM;
}
