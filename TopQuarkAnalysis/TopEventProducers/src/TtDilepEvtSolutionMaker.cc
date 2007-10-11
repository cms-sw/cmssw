//
// $Id: TtDilepEvtSolutionMaker.cc,v 1.8 2007/10/03 22:18:00 lowette Exp $
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
  tauSource_      = iConfig.getParameter<edm::InputTag>("tauSource");
  metSource_      = iConfig.getParameter<edm::InputTag>("metSource");
  jetSource_      = iConfig.getParameter<edm::InputTag>("jetSource");
  nrCombJets_     = iConfig.getParameter<unsigned int> ("nrCombJets");
  matchToGenEvt_  = iConfig.getParameter<bool>         ("matchToGenEvt");
  calcTopMass_    = iConfig.getParameter<bool>         ("calcTopMass"); 
  eeChannel_      = iConfig.getParameter<bool>         ("eeChannel"); 
  emuChannel_     = iConfig.getParameter<bool>         ("emuChannel");
  mumuChannel_    = iConfig.getParameter<bool>         ("mumuChannel");
  mutauChannel_   = iConfig.getParameter<bool>         ("mutauChannel");
  etauChannel_    = iConfig.getParameter<bool>         ("etauChannel");
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
  Handle<std::vector<TopTau> > taus;
  iEvent.getByLabel(tauSource_, taus);
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
  int selTaup = -1, selTaum = -1;
  bool leptonFound = false;

  bool mumu = false;
  bool emu = false;
  bool ee = false;
  bool etau = false;
  bool mutau = false;

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
 else if(muons->size() + electrons->size()==1 && taus->size()>0) {
   // this is the minimal modification of the dilept selection that includes taus.
   // we are considering taus only when no other solution with electrons or muons exist.
   if(muons->size()==1) mutau = true;
   else etau = true;
 }
 
 if(int(ee)+int(emu)+int(mumu)+int(etau)+int(mutau)>1) 
   std::cout << "[TtDilepEvtSolutionMaker]: "
             << "Lepton selection criteria uncorrectly defined" << std::endl;
 
 bool leptonFoundEE = false;
 bool leptonFoundMM = false;
 bool leptonFoundEpMm = false;
 bool leptonFoundEmMp = false;
 bool leptonFoundEpTm = false;
 bool leptonFoundEmTp = false;
 bool leptonFoundMpTm = false;
 bool leptonFoundMmTp = false;
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
  
 else if (mutau) {
   if (LepDiffCharge(&(*taus)[0], &(*muons)[0])) {
     leptonFound = true;
     if (HasPositiveCharge(&(*muons)[0])) {
       leptonFoundMpTm = true;
       selTaum = 0;
       selMuonp = 0;
     }
     else {
       leptonFoundMmTp = true;
       selTaup = 0;
       selMuonm = 0;
     }
   }
 }

 else if (etau) {
   if (LepDiffCharge(&(*taus)[0], &(*electrons)[0])) {
     leptonFound = true;
     if (HasPositiveCharge(&(*electrons)[0])) {
       leptonFoundEpTm = true;
       selTaum = 0;
       selElectronp = 0;
     }
     else {
       leptonFoundEmTp = true;
       selTaup = 0;
       selElectronm = 0;
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
                       (leptonFoundMM && mumuChannel_) ||
		       ((leptonFoundMmTp || leptonFoundMpTm) && mutauChannel_)||
		       ((leptonFoundEmTp || leptonFoundEpTm) && etauChannel_);
                       
  std::vector<TtDilepEvtSolution> * evtsols = new std::vector<TtDilepEvtSolution>();
  if(correctLepton && METFound && jetsFound){

    // protect against reading beyond array boundaries
    unsigned int nrCombJets = nrCombJets_; // do not overwrite nrCombJets_
    if (jets->size() < nrCombJets) nrCombJets = jets->size();

    //SaveSolution for both jet-lep pairings
    for (unsigned int ib = 0; ib < nrCombJets; ib++) {
      TtDilepEvtSolution asol;
      
      double xconstraint = 0, yconstraint = 0;
      if (leptonFoundEE || leptonFoundEpMm || leptonFoundEpTm) {
        asol.setElectronp(electrons, selElectronp);
        xconstraint += (*electrons)[selElectronp].px();
        yconstraint += (*electrons)[selElectronp].py();
      }
      if (leptonFoundEE || leptonFoundEmMp || leptonFoundEmTp) {
        asol.setElectronm(electrons, selElectronm);
        xconstraint += (*electrons)[selElectronm].px();
        yconstraint += (*electrons)[selElectronm].py();
      }
      if (leptonFoundMM || leptonFoundEmMp || leptonFoundMpTm) {
        asol.setMuonp(muons, selMuonp);
        xconstraint += (*muons)[selMuonp].px();
        yconstraint += (*muons)[selMuonp].py();
      }
      if (leptonFoundMM || leptonFoundEpMm || leptonFoundMmTp) {
        asol.setMuonm(muons, selMuonm);
        xconstraint += (*muons)[selMuonm].px();
        yconstraint += (*muons)[selMuonm].py();
      }
      if (leptonFoundEpTm || leptonFoundMpTm) {
        asol.setTaum(taus, selTaum);
        xconstraint += (*taus)[selTaum].px();
        yconstraint += (*taus)[selTaum].py();
      }
      if (leptonFoundEmTp || leptonFoundMmTp) {
        asol.setTaup(taus, selTaup);
        xconstraint += (*taus)[selTaup].px();
        yconstraint += (*taus)[selTaup].py();
      }
      
      if (ib == 0) {asol.setB(jets, 0); asol.setBbar(jets, 1);}
      if (ib == 1) {asol.setB(jets, 1); asol.setBbar(jets, 0);}
      asol.setMET(mets, 0);
      xconstraint += (*jets)[0].px() + (*jets)[1].px() +
                     (*mets)[0].px();
      yconstraint += (*jets)[0].py() + (*jets)[1].py() +
                     (*mets)[0].py();
      
      if (calcTopMass_) {
        Handle<TtGenEvent> genEvent;
        iEvent.getByLabel ("genEvt",genEvent);
  if (genEvent->isFullLeptonic()) {   // FIXME: temporary solution to avoid crash in JetPartonMatching for non semi-leptonic events
        asol.setGenEvt(genEvent);
        TtDilepKinSolver solver(tmassbegin_, tmassend_, tmassstep_, xconstraint, yconstraint);
        asol = solver.addKinSolInfo(&asol);
  }
      }
      
      evtsols->push_back(asol);
    }
    
    // if asked for, match the event solutions to the gen Event
    if(matchToGenEvt_){
      Handle<TtGenEvent> genEvt;
      iEvent.getByLabel ("genEvt",genEvt);
  if (genEvt->isFullLeptonic()) {   // FIXME: temporary solution to avoid crash in JetPartonMatching for non semi-leptonic events
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

