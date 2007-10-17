//
// $Id: TtDilepEvtSolutionMaker.cc,v 1.9 2007/10/11 14:53:46 delaer Exp $
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
  evtSource_      = iConfig.getParameter<edm::InputTag>("evtSource");
  nrCombJets_     = iConfig.getParameter<unsigned int> ("nrCombJets");
  matchToGenEvt_  = iConfig.getParameter<bool>         ("matchToGenEvt");
  calcTopMass_    = iConfig.getParameter<bool>         ("calcTopMass"); 
  useMCforBest_   = iConfig.getParameter<bool>         ("bestSolFromMC");
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

  int selMuonp = -1, selMuonm = -1;
  int selElectronp = -1, selElectronm = -1;
  int selTaup = -1, selTaum = -1;
  bool leptonFound = false;
  bool mumu = false;
  bool emu = false;
  bool ee = false;
  bool etau = false;
  bool mutau = false;
  bool leptonFoundEE = false;
  bool leptonFoundMM = false;
  bool leptonFoundEpMm = false;
  bool leptonFoundEmMp = false;
  bool leptonFoundEpTm = false;
  bool leptonFoundEmTp = false;
  bool leptonFoundMpTm = false;
  bool leptonFoundMmTp = false;
  bool jetsFound = false;
  bool METFound = false;
  int  JetVetoByTaus;
  
  //select MET (TopMET vector is sorted on ET)
  if(mets->size()>=1) { METFound = true; }

  // If we have electrons and muons available, 
  // build a solutions with electrons and muons.
  if (muons->size() + electrons->size() >=2) {
    // select leptons
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
    if (ee) {
      if (LepDiffCharge(&(*electrons)[0], &(*electrons)[1])) {
        leptonFound = true;
        leptonFoundEE = true;
        if (HasPositiveCharge(&(*electrons)[0])) {
          selElectronp = 0;
          selElectronm = 1;
        } else {
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
        } else {
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
        } else {
          selMuonp = 1;
          selMuonm = 0;
        }
      }
    }
    //select Jets (TopJet vector is sorted on ET)
    if(jets->size()>=2) { jetsFound = true; }
  }
  // If a tau is needed to have two leptons, then only consider the taus.
  // This is the minimal modification of the dilept selection that includes taus,
  // since we are considering taus only when no other solution exist.
  else if(muons->size() + electrons->size()==1 && taus->size()>0) {
    // select leptons
    if(muons->size()==1) {
      mutau = true;
      // depending on the muon charge, set the right muon index and specify channel
      int expectedCharge = - muons->begin()->charge();
      int* tauIdx = NULL;
      if (expectedCharge<0) {
	selMuonp = 0;
	tauIdx = &selTaum;
	leptonFoundMpTm = true;
      } else {
	selMuonm = 0;
	tauIdx = &selTaup;
	leptonFoundMmTp = true;
      }
      // loop over the vector of taus to find the first one (highest Pt)
      // that has the charge opposite to the muon one
      for(std::vector<TopTau>::const_iterator tau = taus->begin(); tau < taus->end(); ++tau ) {
        if(tau->charge()*expectedCharge>=0) { *tauIdx = tau-taus->begin(); leptonFound = true; }
      }
      // check that one combination has been found
      if(!leptonFound) { leptonFoundMpTm = false; leptonFoundMmTp = false; } 
      // discard the jet that matches the tau (if one) 
      if(leptonFound) {
        for(std::vector<TopJet>::const_iterator jet = jets->begin(); jet<jets->end() && JetVetoByTaus<0; ++jet) {
          if(DeltaR<reco::Particle>()(*(taus->begin()+*tauIdx),*jet)<0.1) {
            JetVetoByTaus = jet-jets->begin();
          }
	}
      }
    }
    else {
      etau = true;
      // depending on the electron charge, set the right electron index and specify channel
      int expectedCharge = - electrons->begin()->charge();
      int* tauIdx = NULL;
      if (expectedCharge<0) {
	selElectronp = 0;
	tauIdx = &selTaum;
	leptonFoundEpTm = true;
      } else {
	selElectronm = 0;
	tauIdx = &selTaup;
	leptonFoundEmTp = true;
      }
      // loop over the vector of taus to find the first one (highest Pt)
      // that has the charge opposite to the electron one, and does not match in eta-phi
      for(std::vector<TopTau>::const_iterator tau = taus->begin(); tau < taus->end(); ++tau ) {
        if(tau->charge()*expectedCharge>=0 && DeltaR<reco::Particle>()(*tau,*(electrons->begin()))>0.1) { 
	  *tauIdx = tau-taus->begin(); 
	  leptonFound = true; 
	}
      }
      // check that one combination has been found
      if(!leptonFound) { leptonFoundEpTm = false; leptonFoundEmTp = false; } 
      // discard the jet that matches the tau (if one) 
      if(leptonFound) {
        for(std::vector<TopJet>::const_iterator jet = jets->begin(); jet<jets->end() && JetVetoByTaus<0; ++jet) {
          if(DeltaR<reco::Particle>()(*(taus->begin()+*tauIdx),*jet)<0.1) {
            JetVetoByTaus = jet-jets->begin();
          }
        }
      }
    }
    // select Jets (TopJet vector is sorted on ET)
    jetsFound = ((jets->size()>=3)||(jets->size()==2&&JetVetoByTaus<0));
  }
 
  // Check that the above work makes sense
  if(int(ee)+int(emu)+int(mumu)+int(etau)+int(mutau)>1) 
    std::cout << "[TtDilepEvtSolutionMaker]: "
              << "Lepton selection criteria uncorrectly defined" << std::endl;
  
  bool correctLepton = (leptonFoundEE && eeChannel_)                          ||
                       ((leptonFoundEmMp || leptonFoundEpMm) && emuChannel_)  ||
                       (leptonFoundMM && mumuChannel_)                        ||
		       ((leptonFoundMmTp || leptonFoundMpTm) && mutauChannel_)||
		       ((leptonFoundEmTp || leptonFoundEpTm) && etauChannel_)   ;
                       
  std::vector<TtDilepEvtSolution> * evtsols = new std::vector<TtDilepEvtSolution>();
  if(correctLepton && METFound && jetsFound) {
    // protect against reading beyond array boundaries
    unsigned int nrCombJets = nrCombJets_; // do not overwrite nrCombJets_
    if (jets->size() < nrCombJets) nrCombJets = jets->size();
    // consider all permutations
    for (unsigned int ib = 0; ib < nrCombJets; ib++) {
      // skipped jet vetoed during components-flagging.
      if(int(ib)==JetVetoByTaus) continue;
      // second loop of the permutations
      for (unsigned int ibbar = 0; ibbar < nrCombJets; ibbar++) {
        // avoid the diagonal: b and bbar must be distinct jets
        if(ib==ibbar) continue;
	// skipped jet vetoed during components-flagging.
	if(int(ibbar)==JetVetoByTaus) continue;
	// Build and save a solution
        TtDilepEvtSolution asol;
        double xconstraint = 0, yconstraint = 0;
	// Set e+ in the event
        if (leptonFoundEE || leptonFoundEpMm || leptonFoundEpTm) {
          asol.setElectronp(electrons, selElectronp);
          xconstraint += (*electrons)[selElectronp].px();
          yconstraint += (*electrons)[selElectronp].py();
        }
	// Set e- in the event
        if (leptonFoundEE || leptonFoundEmMp || leptonFoundEmTp) {
          asol.setElectronm(electrons, selElectronm);
          xconstraint += (*electrons)[selElectronm].px();
          yconstraint += (*electrons)[selElectronm].py();
        }
	// Set mu+ in the event
        if (leptonFoundMM || leptonFoundEmMp || leptonFoundMpTm) {
          asol.setMuonp(muons, selMuonp);
          xconstraint += (*muons)[selMuonp].px();
          yconstraint += (*muons)[selMuonp].py();
        }
	// Set mu- in the event
        if (leptonFoundMM || leptonFoundEpMm || leptonFoundMmTp) {
          asol.setMuonm(muons, selMuonm);
          xconstraint += (*muons)[selMuonm].px();
          yconstraint += (*muons)[selMuonm].py();
        }
	// Set tau- in the event
        if (leptonFoundEpTm || leptonFoundMpTm) {
          asol.setTaum(taus, selTaum);
          xconstraint += (*taus)[selTaum].px();
          yconstraint += (*taus)[selTaum].py();
        }
	// Set tau+ in the event
        if (leptonFoundEmTp || leptonFoundMmTp) {
          asol.setTaup(taus, selTaup);
          xconstraint += (*taus)[selTaup].px();
          yconstraint += (*taus)[selTaup].py();
        }
	// Set Jets/MET in the event
        asol.setB(jets, ib); 
	asol.setBbar(jets, ibbar);
        asol.setMET(mets, 0);
        xconstraint += (*jets)[ib].px() + (*jets)[ibbar].px() + (*mets)[0].px();
        yconstraint += (*jets)[ib].py() + (*jets)[ibbar].py() + (*mets)[0].py();
	// if asked for, match the event solutions to the gen Event
	if(matchToGenEvt_){
	  Handle<TtGenEvent> genEvt;
	  iEvent.getByLabel (evtSource_,genEvt);
	  asol.setGenEvt(genEvt);
	} 
	// If asked, use the kin fitter to compute the top mass
        if (calcTopMass_) {
          TtDilepKinSolver solver(tmassbegin_, tmassend_, tmassstep_, xconstraint, yconstraint);
	  solver.useWeightFromMC(useMCforBest_);
          asol = solver.addKinSolInfo(&asol);
        }
        evtsols->push_back(asol);
      }
    } 
    // flag the best solution (MC matching)
    if(matchToGenEvt_){
      double bestSolDR = 9999.;
      int bestSol = 0;
      for(size_t s=0; s<evtsols->size(); s++) {
        double dR = (*evtsols)[s].getResidual();
        if(dR<bestSolDR) { bestSolDR = dR; bestSol = s; }
      }
      (*evtsols)[bestSol].setBestSol(true);
    }
    // put the result in the event
    std::auto_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
    iEvent.put(pOut);
  } else {
    // no solution: put a dummy solution in the event
    TtDilepEvtSolution asol;
    evtsols->push_back(asol);
    std::auto_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
    iEvent.put(pOut);
  }
}

