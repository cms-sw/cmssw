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
  explicit MuMuForEmbeddingSelector(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<reco::CompositeCandidate>> ZmumuCandidates_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotLabel_;
  edm::EDGetTokenT<edm::View<pat::MET>> theMETLabel_;
  edm::EDGetTokenT<edm::View<pat::MET>> thePuppiMETLabel_;
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
  produces<edm::RefVector<pat::MuonCollection>>();
  produces<float>("ptLeadingMuon");
  produces<float>("ptTrailingMuon");
  produces<float>("etaLeadingMuon");
  produces<float>("etaTrailingMuon");
  produces<float>("phiLeadingMuon");
  produces<float>("phiTrailingMuon");
  produces<float>("chargeLeadingMuon");
  produces<float>("chargeTrailingMuon");
  produces<float>("dbLeadingMuon");
  produces<float>("dbTrailingMuon");
  produces<float>("massLeadingMuon");
  produces<float>("massTrailingMuon");
  produces<float>("vtxxLeadingMuon");
  produces<float>("vtxyLeadingMuon");
  produces<float>("vtxzLeadingMuon");
  produces<float>("vtxxTrailingMuon");
  produces<float>("vtxyTrailingMuon");
  produces<float>("vtxzTrailingMuon");
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
  for (const auto &vtx : *vertex) {
    if (vtx.isValid() && !vtx.isFake()) {
      posVtx = vtx.position();
      errVtx = vtx.error();
      break;
    }
  }
  reco::Vertex primaryVertex(posVtx, errVtx);

  for (edm::View<reco::CompositeCandidate>::const_iterator iZCand = ZmumuCandidates.begin();
       iZCand != ZmumuCandidates.end();
       ++iZCand) {
    if (chosenZCand == nullptr) {
      chosenZCand = &(*iZCand);
    } else {
      if (iZCand->mass() > chosenZCand->mass()) {
        chosenZCand = &(*iZCand);
      }
    }
  }

  const auto &daughter0 = chosenZCand->daughter(0)->masterClone().castTo<pat::MuonRef>();
  const auto &daughter1 = chosenZCand->daughter(1)->masterClone().castTo<pat::MuonRef>();

  std::unique_ptr<edm::RefVector<pat::MuonCollection>> prod(new edm::RefVector<pat::MuonCollection>());
  prod->reserve(2);
  prod->push_back(daughter0);
  prod->push_back(daughter1);
  iEvent.put(std::move(prod));

  iEvent.put(std::make_unique<float>(daughter0->pt()), "ptLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter1->pt()), "ptTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter0->eta()), "etaLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter1->eta()), "etaTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter0->phi()), "phiLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter1->phi()), "phiTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter0->charge()), "chargeLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter1->charge()), "chargeTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter0->dB()), "dbLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter1->dB()), "dbTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter0->mass()), "massLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter1->mass()), "massTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter0->vertex().x()), "vtxxLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter0->vertex().y()), "vtxyLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter0->vertex().z()), "vtxzLeadingMuon");
  iEvent.put(std::make_unique<float>(daughter1->vertex().x()), "vtxxTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter1->vertex().y()), "vtxyTrailingMuon");
  iEvent.put(std::make_unique<float>(daughter1->vertex().z()), "vtxzTrailingMuon");
  iEvent.put(std::make_unique<bool>(daughter0->isMediumMuon()), "isMediumLeadingMuon");
  iEvent.put(std::make_unique<bool>(daughter0->isTightMuon(primaryVertex)), "isTightLeadingMuon");
  iEvent.put(std::make_unique<bool>(daughter1->isMediumMuon()), "isMediumTrailingMuon");
  iEvent.put(std::make_unique<bool>(daughter1->isTightMuon(primaryVertex)), "isTightTrailingMuon");
  iEvent.put(std::make_unique<float>(met->at(0).et()), "initialMETEt");
  iEvent.put(std::make_unique<float>(met->at(0).phi()), "initialMETphi");
  iEvent.put(std::make_unique<float>(puppimet->at(0).et()), "initialPuppiMETEt");
  iEvent.put(std::make_unique<float>(puppimet->at(0).phi()), "initialPuppiMETphi");
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
