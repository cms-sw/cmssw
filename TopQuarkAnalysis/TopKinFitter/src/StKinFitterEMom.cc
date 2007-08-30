// former parametrisation used in ORCA
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitterEMom.h"


//
// constructor - read in the fit functions from a root file
//
StKinFitterEMom::StKinFitterEMom() {
  maxNrIter = 200;
  maxDeltaS = 5e-5;
  maxF = 1e-4;
  // initialisation
  setupFitter();
}

StKinFitterEMom::StKinFitterEMom(int a, double b, double c, vector<int> d) {
  maxNrIter = a;
  maxDeltaS = b;
  maxF = c;
  constraints = d;
  // initialisation
  setupFitter(); 
}


//
// destructor
//
StKinFitterEMom::~StKinFitterEMom() {
  delete cons1; delete cons2; delete cons3; //delete cons4; delete cons5; delete cons6; delete cons7;
  delete fitBottom; delete fitLight; delete fitLepl; delete fitLepn;
  delete theFitter;
}


StEvtSolution StKinFitterEMom::addKinFitInfo(StEvtSolution * asol) {
  StEvtSolution fitsol(*asol);
  

  TMatrixD m1(4,4), m2(4,4), m3(3,3), m4(3,3);//, m5(3,3), m6(3,3);
  m1.Zero(); m2.Zero(); m3.Zero(); m4.Zero();// m5.Zero(); m6.Zero();
  
  TLorentzVector bottomVec(fitsol.getCalBottom().px(),fitsol.getCalBottom().py(),
                         fitsol.getCalBottom().pz(),fitsol.getCalBottom().energy());
  TLorentzVector lightVec(fitsol.getCalLight().px(),fitsol.getCalLight().py(),
                      	 fitsol.getCalLight().pz(),fitsol.getCalLight().energy());
  TLorentzVector leplVec;
  if(fitsol.getDecay()== "electron") leplVec = TLorentzVector(fitsol.getRecLepe().px(), fitsol.getRecLepe().py(),    
			 				      fitsol.getRecLepe().pz(), fitsol.getRecLepe().energy());
  if(fitsol.getDecay()== "muon")     leplVec = TLorentzVector(fitsol.getRecLepm().px(), fitsol.getRecLepm().py(),    
			 				      fitsol.getRecLepm().pz(), fitsol.getRecLepm().energy());
  TLorentzVector lepnVec(fitsol.getRecLepn().px(), fitsol.getRecLepn().py(),
			 0, fitsol.getRecLepn().et());
 
    
    
// jet resolutions (covM in vector<double> form -> (0,0)=[0], (1,1)=[4], (2,2)=[8])
  m1(0,0) = pow(fitsol.getCalBottom().getResA(),  2);
  m1(1,1) = pow(fitsol.getCalBottom().getResB(), 2);
  m1(2,2) = pow(fitsol.getCalBottom().getResC(), 2);
  m1(3,3) = pow(fitsol.getCalBottom().getResD(), 2);
  m2(0,0) = pow(fitsol.getCalLight().getResA(),  2); 
  m2(1,1) = pow(fitsol.getCalLight().getResB(), 2); 
  m2(2,2) = pow(fitsol.getCalLight().getResC(), 2);
  m2(3,3) = pow(fitsol.getCalLight().getResD(), 2);
  /*
  m3(0,0) = pow(fitsol.getCalHadb().getResA(),  2); 
  m3(1,1) = pow(fitsol.getCalHadb().getResB(), 2); 
  m3(2,2) = pow(fitsol.getCalHadb().getResC(), 2);
  m3(3,3) = pow(fitsol.getCalHadb().getResD(), 2);
  m4(0,0) = pow(fitsol.getCalLepb().getResA(),  2); 
  m4(1,1) = pow(fitsol.getCalLepb().getResB(), 2); 
  m4(2,2) = pow(fitsol.getCalLepb().getResC(), 2);
  m4(3,3) = pow(fitsol.getCalLepb().getResD(), 2);
  */
  if(fitsol.getDecay()== "electron"){
    m3(0,0) = pow(fitsol.getRecLepe().getResA(),  2);
    m3(1,1) = pow(fitsol.getRecLepe().getResB(), 2); 
    m3(2,2) = pow(fitsol.getRecLepe().getResC(), 2);
  }
  if(fitsol.getDecay()== "muon"){
    m3(0,0) = pow(fitsol.getRecLepm().getResA(),  2);
    m3(1,1) = pow(fitsol.getRecLepm().getResB(), 2); 
    m3(2,2) = pow(fitsol.getRecLepm().getResC(), 2);
  }
  m4(0,0) = pow(fitsol.getRecLepn().getResA(),  2);
  m4(1,1) = pow(fitsol.getRecLepn().getResB(),  2);
  m4(2,2) = pow(fitsol.getRecLepn().getResC(), 2);
  
  fitBottom->setIni4Vec(&bottomVec); fitBottom->setCovMatrix(&m1);
  fitLight->setIni4Vec(&lightVec); fitLight->setCovMatrix(&m2);
  fitLepl->setIni4Vec(&leplVec); fitLepl->setCovMatrix(&m3);
  fitLepn->setIni4Vec(&lepnVec); fitLepn->setCovMatrix(&m4);

  theFitter->fit();
  
  // add fitted information to the solution
  if ( theFitter->getStatus() == 0 ) {
    TopParticle aFitBottom(reco::Particle(0,math::XYZTLorentzVector(fitBottom->getCurr4Vec()->X(), fitBottom->getCurr4Vec()->Y(), fitBottom->getCurr4Vec()->Z(), fitBottom->getCurr4Vec()->E()),math::XYZPoint()));
    TopParticle aFitLight(reco::Particle(0,math::XYZTLorentzVector(fitLight->getCurr4Vec()->X(), fitLight->getCurr4Vec()->Y(), fitLight->getCurr4Vec()->Z(), fitLight->getCurr4Vec()->E()),math::XYZPoint()));
    //    TopParticle aFitHadb(reco::Particle(0,math::XYZTLorentzVector(fitHadb->getCurr4Vec()->X(), fitHadb->getCurr4Vec()->Y(), fitHadb->getCurr4Vec()->Z(), fitHadb->getCurr4Vec()->E()),math::XYZPoint()));
    //    TopParticle aFitLepb(reco::Particle(0,math::XYZTLorentzVector(fitLepb->getCurr4Vec()->X(), fitLepb->getCurr4Vec()->Y(), fitLepb->getCurr4Vec()->Z(), fitLepb->getCurr4Vec()->E()),math::XYZPoint()));
    TopParticle aFitLepl(reco::Particle(0,math::XYZTLorentzVector(fitLepl->getCurr4Vec()->X(), fitLepl->getCurr4Vec()->Y(), fitLepl->getCurr4Vec()->Z(), fitLepl->getCurr4Vec()->E()),math::XYZPoint()));
    TopParticle aFitLepn(reco::Particle(0,math::XYZTLorentzVector(fitLepn->getCurr4Vec()->X(), fitLepn->getCurr4Vec()->Y(), fitLepn->getCurr4Vec()->Z(), fitLepn->getCurr4Vec()->E()),math::XYZPoint()));   

    TMatrixD Vb(4,4);  Vb  = (*fitBottom->getCovMatrixFit()); 
    TMatrixD Vq(4,4);  Vq  = (*fitLight->getCovMatrixFit()); 
    aFitBottom.setResA (Vb(0,0));  
    aFitBottom.setResB(Vb(1,1));
    aFitBottom.setResC(Vb(2,2)); 
    aFitBottom.setResD(Vb(3,3)); 
    aFitLight.setResA (Vq(0,0));  
    aFitLight.setResB(Vq(1,1));
    aFitLight.setResC(Vq(2,2));
    aFitLight.setResD(Vq(3,3));
    
    TMatrixD Vl(3,3); Vl = (*fitLepl->getCovMatrixFit()); 
    aFitLepl.setResA (Vl(0,0));  
    aFitLepl.setResB(Vl(1,1));
    aFitLepl.setResC(Vl(2,2));
    
    TMatrixD Vn(3,3); Vn = (*fitLepn->getCovMatrixFit()); 
    aFitLepn.setResA (Vn(0,0));  
    aFitLepn.setResB(Vn(1,1));
    aFitLepn.setResC(Vn(2,2));
    
/*
    TopJet  aFitBottomObj(fitsol.getBottom()); aFitBottomObj.setFitJet(aFitBottom); fitsol.setBottom(aFitBottomObj);
    TopJet  aFitLightObj(fitsol.getLight()); aFitLightObj.setFitJet(aFitLight); fitsol.setLight(aFitLightObj);
    if(fitsol.getDecay() == "muon"){
      TopMuon aFitLeplObj(fitsol.getMuon()); aFitLeplObj.setFitLepton(aFitLepl); fitsol.setMuon(aFitLeplObj);
    }
    else if(fitsol.getDecay() == "electron"){
      TopElectron aFitLeplObj(fitsol.getElectron()); aFitLeplObj.setFitLepton(aFitLepl); fitsol.setElectron(aFitLeplObj);
    }
    TopMET  aFitLepnObj(fitsol.getMET()); aFitLepnObj.setFitMET(aFitLepn); fitsol.setMET(aFitLepnObj);
*/
    fitsol.setChi2(TMath::Prob(theFitter->getS(), theFitter->getNDF()));
  }
  return fitsol;
}


//
// Setup the fitter
//
void StKinFitterEMom::setupFitter() {
  
  theFitter = new TKinFitter("TtFit", "TtFit");

  TMatrixD empty3(3,3); TMatrixD empty4(4,4);
  //  fitHadb = new TFitParticleEMomDev("Jet1", "Jet1", 0, &empty4);
  fitBottom = new TFitParticleEMomDev("Jet1", "Jet1", 0, &empty4);
  fitLight = new TFitParticleEMomDev("Jet2", "Jet2", 0, &empty4);
  //  fitLepb = new TFitParticleEMomDev("Jet4", "Jet4", 0, &empty4);
  fitLepl = new TFitParticleEScaledMomDev("Lepton", "Lepton", 0, &empty3);
  fitLepn = new TFitParticleEScaledMomDev("Neutrino", "Neutrino", 0, &empty3);
  
  //  cons1 = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0 , 80.35);
  //  cons1->addParticles1(fitBottom, fitLight);
  cons1 = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0 , 80.35);
  cons1->addParticles1(fitLepl, fitLepn);
  //  cons3 = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 175.);
  //  cons3->addParticles1(fitBottom, fitLight, fitHadb);
  cons2 = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 175.);
  cons2->addParticles1(fitLepl, fitLepn, fitBottom);
  cons3 = new TFitConstraintM("MassConstraint", "Mass-Constraint", 0, 0, 0.);
  cons3->addParticle1(fitLepn);

  for(unsigned int i=0; i<constraints.size(); i++){
    if(constraints[i] == 1) theFitter->addConstraint(cons1);
    if(constraints[i] == 2) theFitter->addConstraint(cons2);
    if(constraints[i] == 3) theFitter->addConstraint(cons3);
    //    if(constraints[i] == 4) theFitter->addConstraint(cons4);
    //    if(constraints[i] == 5) theFitter->addConstraint(cons5);
  }
  theFitter->addMeasParticle(fitBottom);
  theFitter->addMeasParticle(fitLight);
  theFitter->addMeasParticle(fitLepl);
  theFitter->addMeasParticle(fitLepn);

  theFitter->setMaxNbIter(maxNrIter);
  theFitter->setMaxDeltaS(maxDeltaS);
  theFitter->setMaxF(maxF);
  theFitter->setVerbosity(0);
  
}
