//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopMuonProducer.cc,v 1.8 2007/08/06 14:37:41 tsirig Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <vector>
#include <memory>


//
// constructors and destructor
//

TopMuonProducer::TopMuonProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  muonSrc_        = iConfig.getParameter<edm::InputTag>("muonSource");
  doGenMatch_     = iConfig.getParameter<bool>         ("doGenMatch");
  addResolutions_ = iConfig.getParameter<bool>         ("addResolutions");
  addLRValues_    = iConfig.getParameter<bool>         ("addLRValues");
  genPartSrc_     = iConfig.getParameter<edm::InputTag>("genParticleSource");
  muonResoFile_   = iConfig.getParameter<std::string>  ("muonResoFile");
  muonLRFile_     = iConfig.getParameter<std::string>  ("muonLRFile");

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new TopObjectResolutionCalc(muonResoFile_,iConfig.getParameter<bool>("useNNresolution"));
  }

  // produces vector of muons
  produces<std::vector<TopMuon > >();
}


TopMuonProducer::~TopMuonProducer() {
  if (addResolutions_) delete theResoCalc_;
}


//
// member functions
//

void TopMuonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
 
  // Get the collection of muons from the event
  edm::Handle<std::vector<TopMuonType> > muons;
  iEvent.getByLabel(muonSrc_, muons);

  // Get the vector of generated particles from the event if needed
  edm::Handle<reco::CandidateCollection> particles;
  if (doGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
  }

  // prepare LR calculation if required
  if (addLRValues_) {
    theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, "", muonLRFile_);
  }

  // loop over muons
  std::vector<TopMuon> * topMuons = new std::vector<TopMuon>(); 
  for (size_t m = 0; m < muons->size(); ++m) {
    // construct the TopMuon
    TopMuon aMuon((*muons)[m]);
    // match to generated final state muons
    if (doGenMatch_) {
      // initialize best match as null
      reco::GenParticleCandidate bestGenMuon(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), 0, 0, true);
      float bestDR = 0;
      // find the closest generated muon
      for (reco::CandidateCollection::const_iterator itGenMuon = particles->begin(); itGenMuon != particles->end(); ++itGenMuon) {
        reco::GenParticleCandidate aGenMuon = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(&*itGenMuon)));
        if (abs(aGenMuon.pdgId())==13 && aGenMuon.status()==1 && aGenMuon.charge()==(*muons)[m].charge()) {
          float currDR = DeltaR<reco::Candidate>()(aGenMuon, (*muons)[m]);
          if (bestDR == 0 || currDR < bestDR) {
            bestGenMuon = aGenMuon;
            bestDR = currDR;
          }
        }
        aMuon.setGenLepton(bestGenMuon);
      }
    }
    // add resolution info if demanded
    if (addResolutions_) {
      (*theResoCalc_)(aMuon);
    }
    // add top lepton id LR info if requested
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(aMuon, iEvent);
    }
    // add the muon to the vector of TopMuons
    topMuons->push_back(TopMuon(aMuon));
  }

  // sort muons in pT
  std::sort(topMuons->begin(), topMuons->end(), pTMuonComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<TopMuon> > pOutMuon(topMuons);
  iEvent.put(pOutMuon);

  // destroy the lepton LR calculator
  if (addLRValues_) delete theLeptonLRCalc_;

}
