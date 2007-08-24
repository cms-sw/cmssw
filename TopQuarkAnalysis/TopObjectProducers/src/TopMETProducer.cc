//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopMETProducer.cc,v 1.5.2.1 2007/08/24 13:52:27 delaer Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMETProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <memory>


//
// constructors and destructor
//

TopMETProducer::TopMETProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  metSrc_         = iConfig.getParameter<edm::InputTag>("metSource");
  calcGenMET_     = iConfig.getParameter<bool>         ("calcGenMET");
  addResolutions_ = iConfig.getParameter<bool>         ("addResolutions");
  addMuonCorr_    = iConfig.getParameter<bool>         ("addMuonCorrections");
  genPartSrc_     = iConfig.getParameter<edm::InputTag>("genParticleSource");
  metResoFile_    = iConfig.getParameter<std::string>  ("metResoFile");
  muonSrc_        = iConfig.getParameter<edm::InputTag>("muonSource");   
  
  // construct resolution calculator
  if(addResolutions_){
    metResoCalc_ = new TopObjectResolutionCalc(metResoFile_,iConfig.getParameter<bool>("useNNresolution"));
  }
  
  // produces vector of mets
  produces<std::vector<TopMET> >();
}


TopMETProducer::~TopMETProducer() {
  if (addResolutions_) delete metResoCalc_;
}


//
// member functions
//

void TopMETProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of MET's from the event
  edm::Handle<std::vector<TopMETType> > mets;
  iEvent.getByLabel(metSrc_, mets);

  // Get the vector of generated particles from the event if needed
  edm::Handle<reco::CandidateCollection> particles;
  if (calcGenMET_) {
    iEvent.getByLabel(genPartSrc_, particles);
  }

  // read in the muons if demanded
  edm::Handle<std::vector<TopMuonType> > muons;
  if (addMuonCorr_) {
    iEvent.getByLabel(muonSrc_, muons);
  }
  
  // loop over mets
  std::vector<TopMET> * ttMETs = new std::vector<TopMET>(); 
  for (size_t j = 0; j < mets->size(); j++) {
    // construct the TopMET
    TopMET amet((*mets)[j]);
    // calculate the generated MET (just sum of neutrinos)
    if (calcGenMET_) {
      reco::Particle theGenMET(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0));
      for(reco::CandidateCollection::const_iterator itGenPart = particles->begin(); itGenPart != particles->end(); ++itGenPart) {
        reco::Candidate * aTmpGenPart = const_cast<reco::Candidate *>(&*itGenPart);
        reco::GenParticleCandidate aGenPart = *(dynamic_cast<reco::GenParticleCandidate *>(aTmpGenPart));
        if ((aGenPart.status()==1) &&
            (abs(aGenPart.pdgId())==12 || abs(aGenPart.pdgId())==14 || abs(aGenPart.pdgId())==16)) {
          theGenMET.setP4(theGenMET.p4() + aGenPart.p4());
        }
      }
      amet.setGenMET(theGenMET);
    }
    // add MET resolution info if demanded
    if (addResolutions_) {
      (*metResoCalc_)(amet);
    }
    // correct for muons if demanded
    if (addMuonCorr_) {
      for (size_t m = 0; m < muons->size(); m++) {
        amet.setP4(reco::Particle::LorentzVector(
            amet.px()-(*muons)[m].px(),
            amet.py()-(*muons)[m].py(),
            0,
            sqrt(pow(amet.px()-(*muons)[m].px(), 2)+pow(amet.py()-(*muons)[m].py(), 2))
        ));
      }
    }
    // add the MET to the vector of TopMETs
    ttMETs->push_back(TopMET(amet));
  }

  // sort MET in ET
  std::sort(ttMETs->begin(), ttMETs->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<TopMET> > myTopMETProducer(ttMETs);
  iEvent.put(myTopMETProducer);

}
