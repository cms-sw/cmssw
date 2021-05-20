// -*- C++ -*-
//
// Package:    MyTemporarySubSystem/PackedCandidateGenAssociationProducer
// Class:      PackedCandidateGenAssociationProducer
//
/**\class PackedCandidateGenAssociationProducer PackedCandidateGenAssociationProducer.cc MyTemporarySubSystem/PackedCandidateGenAssociationProducer/plugins/PackedCandidateGenAssociationProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Enrico Lusiani
//         Created:  Mon, 03 May 2021 13:40:39 GMT
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

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

//
// class declaration
//

class PackedCandidateGenAssociationProducer : public edm::stream::EDProducer<> {
public:
  explicit PackedCandidateGenAssociationProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> trackToGenToken_;
  edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> trackToPcToken_;
  edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> genToPrunedToken_;
  edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> genToPrunedWSOToken_;
  edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;
};
PackedCandidateGenAssociationProducer::PackedCandidateGenAssociationProducer(const edm::ParameterSet& iConfig)
    : trackToGenToken_(consumes<edm::Association<reco::GenParticleCollection>>(
          iConfig.getParameter<edm::InputTag>("trackToGenAssoc"))),
      trackToPcToken_(consumes<edm::Association<pat::PackedCandidateCollection>>(
          iConfig.getParameter<edm::InputTag>("trackToPackedCandidatesAssoc"))),
      genToPrunedToken_(consumes<edm::Association<reco::GenParticleCollection>>(
          iConfig.getParameter<edm::InputTag>("genToPrunedAssoc"))),
      genToPrunedWSOToken_(consumes<edm::Association<reco::GenParticleCollection>>(
          iConfig.getParameter<edm::InputTag>("genToPrunedAssocWithStatusOne"))),
      tracksToken_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("tracks"))) {
  produces<edm::Association<reco::GenParticleCollection>>();
}

void PackedCandidateGenAssociationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<edm::Association<reco::GenParticleCollection>> htrackToGenAssoc;
  iEvent.getByToken(trackToGenToken_, htrackToGenAssoc);
  const auto& trackToGenAssoc = *htrackToGenAssoc;
  auto recoGenCollection = trackToGenAssoc.ref();

  edm::Handle<edm::Association<pat::PackedCandidateCollection>> htrackToPackedCandidatesAssoc;
  iEvent.getByToken(trackToPcToken_, htrackToPackedCandidatesAssoc);
  const auto& trackToPackedCandidatesAssoc = *htrackToPackedCandidatesAssoc;
  auto pcCollection = trackToPackedCandidatesAssoc.ref();

  edm::Handle<edm::Association<reco::GenParticleCollection>> hgenToPrunedAssoc;
  iEvent.getByToken(genToPrunedToken_, hgenToPrunedAssoc);
  const auto& genToPrunedAssoc = *hgenToPrunedAssoc;
  auto prunedCollection = genToPrunedAssoc.ref();

  edm::Handle<edm::Association<reco::GenParticleCollection>> hgenToPrunedAssocWSO;
  iEvent.getByToken(genToPrunedWSOToken_, hgenToPrunedAssocWSO);
  const auto& genToPrunedAssocWSO = *hgenToPrunedAssocWSO;

  edm::Handle<edm::View<reco::Track>> htracks;
  iEvent.getByToken(tracksToken_, htracks);
  const auto& tracks = *htracks;

  auto out = std::make_unique<edm::Association<reco::GenParticleCollection>>(prunedCollection);
  Association<reco::GenParticleCollection>::Filler filler(*out);

  std::vector<int> indices(pcCollection->size(), -1);

  for (size_t i = 0; i < tracks.size(); i++) {
    auto track = tracks.refAt(i);

    auto pc = trackToPackedCandidatesAssoc[track];
    if (pc.isNull()) {
      continue;
    }

    auto gen = trackToGenAssoc[track];
    if (gen.isNull()) {
      continue;
    }

    auto newGenWSO = genToPrunedAssocWSO[gen];
    if (newGenWSO.isNull()) {
      continue;
    }

    auto newGen = genToPrunedAssoc[newGenWSO];
    if (newGen.isNull()) {
      continue;
    }

    indices[pc.index()] = newGen.index();
  }
  filler.insert(pcCollection, indices.begin(), indices.end());
  filler.fill();
  iEvent.put(std::move(out));
}

void PackedCandidateGenAssociationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genToPrunedAssoc", edm::InputTag("prunedGenParticles"));
  desc.add<edm::InputTag>("genToPrunedAssocWithStatusOne", edm::InputTag("prunedGenParticlesWithStatusOne"));
  desc.add<edm::InputTag>("trackToPackedCandidatesAssoc", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("trackToGenAssoc");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));

  descriptions.add("packedCandidatesGenAssociation", desc);
}

DEFINE_FWK_MODULE(PackedCandidateGenAssociationProducer);
