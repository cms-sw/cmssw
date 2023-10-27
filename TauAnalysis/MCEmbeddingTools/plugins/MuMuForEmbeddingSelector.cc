// -*- C++ -*-
//
// Package:    TauAnalysis/EmbeddingProducer
// Class:      MuMuForEmbeddingSelector
//
/**\class MuMuForEmbeddingSelector MuMuForEmbeddingSelector.cc TauAnalysis/EmbeddingProducer/plugins/MuMuForEmbeddingSelector.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Artur Akhmetshin
//         Created:  Mon, 13 Jun 2016 11:05:32 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

//
// class declaration
//

class MuMuForEmbeddingSelector : public edm::stream::EDProducer<> {
public:
  explicit MuMuForEmbeddingSelector(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<reco::CompositeCandidate>> ZmumuCandidates_;
  double ZMass = 91.0;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuMuForEmbeddingSelector::MuMuForEmbeddingSelector(const edm::ParameterSet& iConfig)
    : ZmumuCandidates_(consumes<edm::View<reco::CompositeCandidate>>(
          iConfig.getParameter<edm::InputTag>("ZmumuCandidatesCollection"))) {
  //register your products
  /* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
 
   //if you want to put into the Run
   produces<ExampleData2,InRun>();
*/
  produces<edm::RefVector<pat::MuonCollection>>();

  //now do what ever other initialization is needed
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MuMuForEmbeddingSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::Handle<edm::View<reco::CompositeCandidate>> ZmumuCandidatesHandle;
  iEvent.getByToken(ZmumuCandidates_, ZmumuCandidatesHandle);
  edm::View<reco::CompositeCandidate> ZmumuCandidates = *ZmumuCandidatesHandle;

  const reco::CompositeCandidate* chosenZCand = nullptr;
  double massDifference = -1.0;
  for (edm::View<reco::CompositeCandidate>::const_iterator iZCand = ZmumuCandidates.begin();
       iZCand != ZmumuCandidates.end();
       ++iZCand) {
    if (std::abs(ZMass - iZCand->mass()) < massDifference || massDifference < 0) {
      massDifference = std::abs(ZMass - iZCand->mass());
      chosenZCand = &(*iZCand);
    }
  }
  std::unique_ptr<edm::RefVector<pat::MuonCollection>> prod(new edm::RefVector<pat::MuonCollection>());
  prod->reserve(2);
  prod->push_back(chosenZCand->daughter(0)->masterClone().castTo<pat::MuonRef>());
  prod->push_back(chosenZCand->daughter(1)->masterClone().castTo<pat::MuonRef>());
  iEvent.put(std::move(prod));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MuMuForEmbeddingSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuMuForEmbeddingSelector);
