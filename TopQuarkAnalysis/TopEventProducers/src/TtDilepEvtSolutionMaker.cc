//
// $Id$
//

#include "TopQuarkAnalysis/TopEventProducers/interface/TtDilepEvtSolutionMaker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtDilepKinSolver.h"

#include <memory>


/// constructor
TtDilepEvtSolutionMaker::TtDilepEvtSolutionMaker(const edm::ParameterSet & iConfig) {
  // configurables
  electronSource_ = iConfig.getParameter<edm::InputTag>("electronSource");
  muonSource_     = iConfig.getParameter<edm::InputTag>("muonSource");
  metSource_      = iConfig.getParameter<edm::InputTag>("metSource");
  jetSource_      = iConfig.getParameter<edm::InputTag>("jetSource");
  matchToGenEvt_  = iConfig.getParameter<bool>         ("matchToGenEvt");
  calcTopMass_    = iConfig.getParameter<bool>         ("calcTopMass"); 
  eeChannel_      = iConfig.getParameter<bool>         ("eeChannel"); 
  emuChannel_     = iConfig.getParameter<bool>         ("emuChannel");
  mumuChannel_    = iConfig.getParameter<bool>         ("mumuChannel");
  tmassbegin_     = iConfig.getParameter<double>       ("tmassbegin");
  tmassend_       = iConfig.getParameter<double>       ("tmassend");
  tmassstep_      = iConfig.getParameter<double>       ("tmassstep");
  
  // define what will be produced
  produces<std::vector<TtDilepEvtSolution> >();
}


/// destructor
TtDilepEvtSolutionMaker::~TtDilepEvtSolutionMaker() {
}



void TtDilepEvtSolutionMaker::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  using namespace edm;
  Handle<std::vector<TopMuon> > muons;
  iEvent.getByLabel(muonSource_, muons);
  Handle<std::vector<TopElectron> > electrons;
  iEvent.getByLabel(electronSource_, electrons);
  Handle<std::vector<TopMET> > mets;
  iEvent.getByLabel(metSource_, mets);
  Handle<std::vector<TopJet> > jets;
  iEvent.getByLabel(jetSource_, jets);

  // select lepton (the TtLepton vectors are, for the moment, sorted on pT)
  int selMuonp = -1, selMuonm = -1;
  int selElectronp = -1, selElectronm = -1;
  bool leptonFound = false;

  bool mumu = false;
  bool emu = false;
  bool ee = false;

  if (muons->size() + electrons->size() >=2) {
    if (electrons->size() == 0) mumu = true;
    
    else if (muons->size() == 0) ee = true;
    
    else if (electrons->size() == 1) {
      if (muons->size() == 1) emu = true;
      else if (PTComp(&(*electrons)[0], &(*muons)[1])) emu = true;
      else  mumu = true;
    }
    
    else if (electrons->size() > 1) {
      if (PTComp(&(*electrons)[1], &(*muons)[0])) ee = true;
      else if (muons->size() == 1) emu = true;
      else if (PTComp(&(*electrons)[0], &(*muons)[1])) emu = true;
      else mumu = true;
    }
 }
 
 if ((ee && emu) || (ee && mumu) || (emu && mumu))
   std::cout << "[TtDilepEvtSolutionMaker]: "
        << "Lepton selection criteria uncorrectly defined" << std::endl;
 
 bool leptonFoundEE = false;
 bool leptonFoundMM = false;
 bool leptonFoundEpMm = false;
 bool leptonFoundEmMp = false;
 if (ee) {
   if (LepDiffCharge(&(*electrons)[0], &(*electrons)[1])) {
     leptonFound = true;
     leptonFoundEE = true;
     if (HasPositiveCharge(&(*electrons)[0])) {
       selElectronp = 0;
       selElectronm = 1;
     }
     else {
       selElectronp = 1;
       selElectronm = 0;
     }
   }
 }
 
 else if (emu) {
   if (LepDiffCharge(&(*electrons)[0], &(*muons)[0])) {
     leptonFound = true;
     if (HasPositiveCharge(&(*electrons)[0])) {
       leptonFoundEpMm = true;
       selElectronp = 0;
       selMuonm = 0;
     }
     else {
       leptonFoundEmMp = true;
       selMuonp = 0;
       selElectronm = 0;
     }
   }
 }
 
 else if (mumu) {
   if (LepDiffCharge(&(*muons)[0], &(*muons)[1])) {
     leptonFound = true;
     leptonFoundMM = true;
     if (HasPositiveCharge(&(*muons)[0])) {
       selMuonp = 0;
       selMuonm = 1;
     }
     else {
       selMuonp = 1;
       selMuonm = 0;
     }
   }
 }
  
  //select MET (TopMET vector is sorted on ET)
  bool METFound = false;
  if(mets -> size()>=1) { METFound = true; }

  //select Jets (TopJet vector is sorted on ET)
  bool jetsFound = false;
  if(jets -> size()>=2) { jetsFound = true; }
  
  bool correctLepton = (leptonFoundEE && eeChannel_) ||
                       ((leptonFoundEmMp || leptonFoundEpMm) && emuChannel_) ||
                       (leptonFoundMM && mumuChannel_);
                       
  std::vector<TtDilepEvtSolution> * evtsols = new std::vector<TtDilepEvtSolution>();
  if(correctLepton && METFound && jetsFound){
    //cout<<"constructing solutions"<<endl;
    
    //SaveSolution for both jet-lep pairings
    for (unsigned int ib = 0; ib < 2; ib++) {
      TtDilepEvtSolution asol;
      
      double xconstraint = 0, yconstraint = 0;
      if (leptonFoundEE || leptonFoundEpMm) {
        asol.setElectronp(electrons, selElectronp);
        xconstraint += (*electrons)[selElectronp].px();
        yconstraint += (*electrons)[selElectronp].py();
      }
      if (leptonFoundEE || leptonFoundEmMp) {
        asol.setElectronm(electrons, selElectronm);
        xconstraint += (*electrons)[selElectronm].px();
        yconstraint += (*electrons)[selElectronm].py();
      }
      if (leptonFoundMM || leptonFoundEmMp) {
        asol.setMuonp(muons, selMuonp);
        xconstraint += (*muons)[selMuonp].px();
        yconstraint += (*muons)[selMuonp].py();
      }
      if (leptonFoundMM || leptonFoundEpMm) {
        asol.setMuonm(muons, selMuonm);
        xconstraint += (*muons)[selMuonm].px();
        yconstraint += (*muons)[selMuonm].py();
      }
      
      if (ib == 0) {asol.setB(jets, 0); asol.setBbar(jets, 1);}
      if (ib == 1) {asol.setB(jets, 1); asol.setBbar(jets, 0);}
      asol.setMET(mets, 0);
      xconstraint += (*jets)[0].px() + (*jets)[1].px() +
                     (*mets)[0].px();
      yconstraint += (*jets)[0].py() + (*jets)[1].py() +
                     (*mets)[0].py();
      
      if (calcTopMass_) {
        TtDilepKinSolver solver(tmassbegin_, tmassend_, tmassstep_, xconstraint, yconstraint);
        asol = solver.addKinSolInfo(&asol);
      }
      
      evtsols->push_back(asol);
    }
    
    // if asked for, match the event solutions to the gen Event
    if(matchToGenEvt_){
      Handle<TtGenEvent> genEvt;
      iEvent.getByLabel ("genEvt",genEvt);
      double bestSolDR = 9999.;
      int bestSol = 0;
      for(size_t s=0; s<evtsols->size(); s++) {
        (*evtsols)[s].setGenEvt(genEvt);
        //FIXME probably this should be moved to BestMatching.h
        double dRBB =       DeltaR<reco::Particle>()((reco::Particle) (*evtsols)[s].getCalJetB(), (reco::Particle) *((*evtsols)[s].getGenB()));
        double dRBbarBbar = DeltaR<reco::Particle>()((reco::Particle) (*evtsols)[s].getCalJetBbar(), (reco::Particle) *((*evtsols)[s].getGenBbar()));
        if (dRBB + dRBbarBbar < bestSolDR) { bestSolDR = dRBB + dRBbarBbar; bestSol = s; }
      }
      (*evtsols)[bestSol].setBestSol(true);
    }
    
    std::auto_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
    iEvent.put(pOut);
  }
  else {
    TtDilepEvtSolution asol;
    evtsols->push_back(asol);
    std::auto_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
    iEvent.put(pOut);
  }

}

bool TtDilepEvtSolutionMaker::PTComp(const TopElectron * e, const TopMuon * m) const {
  if (e->pt() > m->pt()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::LepDiffCharge(const TopElectron * e, const TopMuon * m) const {
  if (e->charge() != m->charge()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::LepDiffCharge(const TopElectron * e1, const TopElectron * e2) const {
  if (e1->charge() != e2->charge()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::LepDiffCharge(const TopMuon * m1, const TopMuon * m2) const {
  if (m1->charge() != m2->charge()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::HasPositiveCharge(const TopMuon * m) const {
  if (m->charge() > 0) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::HasPositiveCharge(const TopElectron * e) const {
  if (e->charge() > 0) return true;
  else return false;
}

