//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id$
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <vector>
#include <memory>


//
// constructors and destructor
//

TopJetProducer::TopJetProducer(const edm::ParameterSet& iConfig) {
  // initialize the configurables
  jetTagsLabel_    = iConfig.getParameter<edm::InputTag>("jetTagInput");
  recJetsLabel_    = iConfig.getParameter<edm::InputTag>("recJetInput");
  caliJetsLabel_   = iConfig.getParameter<edm::InputTag>("caliJetInput");
  addResolutions_  = iConfig.getParameter<bool>         ("addResolutions");
  caliJetResoFile_ = iConfig.getParameter<std::string>  ("caliJetResoFile");

  // construct resolution calculator
  if (addResolutions_) theResoCalc_ = new TopObjectResolutionCalc(caliJetResoFile_);

  // produces vector of jets
  produces<std::vector<TopJet> >();
}


TopJetProducer::~TopJetProducer() {
  if(addResolutions_) delete theResoCalc_;
}


//
// member functions
//

void TopJetProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of generated particles from the event
  edm::Handle<std::vector<JetType> > recjets;
  iEvent.getByLabel(recJetsLabel_, recjets);
  edm::Handle<std::vector<JetType> > calijets;
  iEvent.getByLabel(caliJetsLabel_, calijets);
  edm::Handle<std::vector<reco::JetTag> > jetTags;
  iEvent.getByLabel(jetTagsLabel_, jetTags);

  // loop over jets
  std::vector<TopJet> * topJets = new std::vector<TopJet>(); 
  for (size_t j = 0; j < recjets->size(); j++) {
    // construct the TopJet
    TopJet ajet;
    // loop over cal jets to find corresponding jet
    bool cjFound = false;
    for (size_t cj = 0; cj < calijets->size(); cj++) {
      // FIXME: is this 0.01 matching fullproof?
      if (DeltaR<reco::Candidate>()((*recjets)[j], (*calijets)[cj]) < 0.01) {
        cjFound = true;
        ajet = TopJet((*calijets)[cj]);
        ajet.setRecJet((*recjets)[j]);
      }
    }
    // if cal jet found...
    if (cjFound) {
      // add b-tag info if available
      for (size_t t = 0; t < jetTags->size(); t++) {
        // FIXME: is this 0.0001 matching fullproof?
        if (DeltaR<reco::Candidate>()((*recjets)[j], (*jetTags)[t].jet()) < 0.0001) {
          ajet.setBdiscriminant((*jetTags)[t].discriminator());
        }
      }
      // add resolution info if demanded
      if (addResolutions_) {
        (*theResoCalc_)(ajet);
      }
    } else {
      std::cout << "no cal jet found " << std::endl;
    }
    topJets->push_back(ajet);
  }

  // sort jets in ET
  std::sort(topJets->begin(), topJets->end(), eTComparator_);

  // put genEvt  in Event
  std::auto_ptr<std::vector<TopJet> > myTopJetProducer(topJets);
  iEvent.put(myTopJetProducer);

}
