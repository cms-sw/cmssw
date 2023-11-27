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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/PatCandidates/interface/MET.h"

//
// class declaration
//

class MuMuForEmbeddingSelector : public edm::stream::EDProducer<> {
public:
  explicit MuMuForEmbeddingSelector(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<reco::CompositeCandidate>> ZmumuCandidates_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotLabel_;
  edm::EDGetTokenT<edm::View<pat::MET>> theMETLabel_;
  edm::EDGetTokenT<edm::View<pat::MET>> thePuppiMETLabel_;
  bool use_zmass = false;
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
MuMuForEmbeddingSelector::MuMuForEmbeddingSelector(const edm::ParameterSet &iConfig)
    : ZmumuCandidates_(consumes<edm::View<reco::CompositeCandidate>>(
          iConfig.getParameter<edm::InputTag>("ZmumuCandidatesCollection"))) {
  use_zmass = iConfig.getParameter<bool>("use_zmass");
  produces<edm::RefVector<pat::MuonCollection>>();
  produces<float>("oldMass");
  produces<float>("newMass");
  produces<float>("nPairCandidates");
  produces<bool>("isMediumLeadingMuon");
  produces<bool>("isTightLeadingMuon");
  produces<bool>("isMediumTrailingMuon");
  produces<bool>("isTightTrailingMuon");
  produces<float>("initialMETEt");
  produces<float>("initialMETphi");
  produces<float>("initialPuppiMETEt");
  produces<float>("initialPuppiMETphi");
  theVertexLabel_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("inputTagVertex"));
  theBeamSpotLabel_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("inputTagBeamSpot"));
  theMETLabel_ = consumes<edm::View<pat::MET>>(iConfig.getParameter<edm::InputTag>("Met"));
  thePuppiMETLabel_ = consumes<edm::View<pat::MET>>(iConfig.getParameter<edm::InputTag>("PuppiMet"));
  // now do what ever other initialization is needed
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MuMuForEmbeddingSelector::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  edm::Handle<edm::View<reco::CompositeCandidate>> ZmumuCandidatesHandle;
  iEvent.getByToken(ZmumuCandidates_, ZmumuCandidatesHandle);
  edm::View<reco::CompositeCandidate> ZmumuCandidates = *ZmumuCandidatesHandle;
  const reco::CompositeCandidate *chosenZCand = nullptr;
  const reco::CompositeCandidate *chosenZCand_zmass = nullptr;
  const reco::CompositeCandidate *chosenZCand_largest = nullptr;
  double massDifference = -1.0;
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(theBeamSpotLabel_, beamSpot);
  edm::Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(theVertexLabel_, vertex);
  edm::Handle<edm::View<pat::MET>> met;
  iEvent.getByToken(theMETLabel_, met);
  edm::Handle<edm::View<pat::MET>> puppimet;
  iEvent.getByToken(thePuppiMETLabel_, puppimet);
  // get primary vertex
  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  std::vector<reco::Vertex>::const_iterator vertexIt = vertex->begin();
  std::vector<reco::Vertex>::const_iterator vertexEnd = vertex->end();
  for (; vertexIt != vertexEnd; ++vertexIt) {
    if (vertexIt->isValid() && !vertexIt->isFake()) {
      posVtx = vertexIt->position();
      errVtx = vertexIt->error();
      break;
    }
  }
  reco::Vertex primaryVertex(posVtx, errVtx);

  for (edm::View<reco::CompositeCandidate>::const_iterator iZCand = ZmumuCandidates.begin();
       iZCand != ZmumuCandidates.end();
       ++iZCand) {
    if (std::abs(ZMass - iZCand->mass()) < massDifference || massDifference < 0) {
      massDifference = std::abs(ZMass - iZCand->mass());
      chosenZCand_zmass = &(*iZCand);
    }
  }
  for (edm::View<reco::CompositeCandidate>::const_iterator iZCand = ZmumuCandidates.begin();
       iZCand != ZmumuCandidates.end();
       ++iZCand) {
    if (chosenZCand_largest == nullptr) {
      chosenZCand_largest = &(*iZCand);
    } else {
      if (iZCand->mass() > chosenZCand_largest->mass()) {
        chosenZCand_largest = &(*iZCand);
      }
    }
  }
  if (use_zmass) {
    // edm::LogDebug("MuMuForEmbeddingSelector") << "Using Z mass candidate" << chosenZCand_zmass->mass();
    chosenZCand = chosenZCand_zmass;
  } else {
    // edm::LogDebug("MuMuForEmbeddingSelector") << "Using largest mass candidate" << chosenZCand_largest->mass();
    chosenZCand = chosenZCand_largest;
  }

  std::unique_ptr<edm::RefVector<pat::MuonCollection>> prod(new edm::RefVector<pat::MuonCollection>());
  prod->reserve(2);
  prod->push_back(chosenZCand->daughter(0)->masterClone().castTo<pat::MuonRef>());
  prod->push_back(chosenZCand->daughter(1)->masterClone().castTo<pat::MuonRef>());
  iEvent.put(std::move(prod));
  iEvent.put(std::make_unique<float>(chosenZCand_zmass->mass()), "oldMass");
  iEvent.put(std::make_unique<float>(chosenZCand_largest->mass()), "newMass");
  iEvent.put(std::make_unique<float>(ZmumuCandidates.size()), "nPairCandidates");
  iEvent.put(std::make_unique<bool>(chosenZCand->daughter(0)->masterClone().castTo<pat::MuonRef>()->isMediumMuon()),
             "isMediumLeadingMuon");
  iEvent.put(std::make_unique<bool>(
                 chosenZCand->daughter(0)->masterClone().castTo<pat::MuonRef>()->isTightMuon(primaryVertex)),
             "isTightLeadingMuon");
  iEvent.put(std::make_unique<bool>(chosenZCand->daughter(1)->masterClone().castTo<pat::MuonRef>()->isMediumMuon()),
             "isMediumTrailingMuon");
  iEvent.put(std::make_unique<bool>(
                 chosenZCand->daughter(1)->masterClone().castTo<pat::MuonRef>()->isTightMuon(primaryVertex)),
             "isTightTrailingMuon");
  iEvent.put(std::make_unique<float>(met->at(0).et()), "initialMETEt");
  iEvent.put(std::make_unique<float>(met->at(0).phi()), "initialMETphi");
  iEvent.put(std::make_unique<float>(puppimet->at(0).et()), "initialPuppiMETEt");
  iEvent.put(std::make_unique<float>(puppimet->at(0).phi()), "initialPuppiMETphi");
  // edm::LogDebug("MuMuForEmbeddingSelector") << "PuppiMet: " << puppimet->at(0).et() << " phi: " << puppimet->at(0).phi();
  // edm::LogDebug("MuMuForEmbeddingSelector") << "MET: " << met->at(0).et() << " phi: " << met->at(0).phi();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MuMuForEmbeddingSelector::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  //  Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuMuForEmbeddingSelector);
