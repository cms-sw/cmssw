// -*- C++ -*-
//
// Package:    SimTracker/TrackAssociation
// Class:      DigiSimLinkPruner
//
/**\class DigiSimLinkPruner DigiSimLinkPruner.cc SimTracker/TrackAssociation/plugins/DigiSimLinkPruner.cc

 Description: Produce a pruned version of the DigiSimLinks collection based on the association to a collection of TrackingParticles

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Enrico Lusiani
//         Created:  Fri, 14 May 2021 08:46:10 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/Track/interface/UniqueSimTrackId.h"

#include <unordered_set>

namespace {

  template <typename T>
  std::vector<edm::DetSet<T>> pruneByTpAssoc(
      const edm::DetSetVector<T>& simLinkColl,
      const std::unordered_set<UniqueSimTrackId, UniqueSimTrackIdHash>& selectedIds) {
    std::vector<edm::DetSet<T>> linkVector;

    for (auto&& detSet : simLinkColl) {
      edm::DetSet<T> newDetSet(detSet.detId());

      for (auto&& simLink : detSet) {
        UniqueSimTrackId trkid(simLink.SimTrackId(), simLink.eventId());
        if (selectedIds.count(trkid) > 0) {
          newDetSet.push_back(simLink);
        }
      }

      linkVector.push_back(std::move(newDetSet));
    }

    return linkVector;
  }

}  // namespace

//
// class declaration
//

class DigiSimLinkPruner : public edm::global::EDProducer<> {
public:
  explicit DigiSimLinkPruner(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> sipixelSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink>> sistripSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> siphase2OTSimLinksToken_;
};

DigiSimLinkPruner::DigiSimLinkPruner(const edm::ParameterSet& iConfig)
    : trackingParticleToken_(
          consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))),
      sipixelSimLinksToken_(
          consumes<edm::DetSetVector<PixelDigiSimLink>>(iConfig.getParameter<edm::InputTag>("pixelSimLinkSrc"))) {
  produces<edm::DetSetVector<PixelDigiSimLink>>("siPixel");

  if (iConfig.existsAs<edm::InputTag>("stripSimLinkSrc")) {
    sistripSimLinksToken_ =
        consumes<edm::DetSetVector<StripDigiSimLink>>(iConfig.getParameter<edm::InputTag>("stripSimLinkSrc"));
    produces<edm::DetSetVector<StripDigiSimLink>>("siStrip");
  }

  if (iConfig.existsAs<edm::InputTag>("phase2OTSimLinkSrc")) {
    siphase2OTSimLinksToken_ =
        consumes<edm::DetSetVector<PixelDigiSimLink>>(iConfig.getParameter<edm::InputTag>("phase2OTSimLinkSrc"));
    produces<edm::DetSetVector<PixelDigiSimLink>>("siphase2OT");
  }
}

// ------------ method called to produce the data  ------------
void DigiSimLinkPruner::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  auto const& tpColl = iEvent.get(trackingParticleToken_);

  std::unordered_set<UniqueSimTrackId, UniqueSimTrackIdHash> selectedIds;
  for (TrackingParticleCollection::size_type itp = 0; itp < tpColl.size(); ++itp) {
    auto const& trackingParticle = tpColl[itp];

    // SimTracks inside TrackingParticle

    EncodedEventId eid(trackingParticle.eventId());

    for (auto const& trk : trackingParticle.g4Tracks()) {
      selectedIds.emplace(trk.trackId(), eid);
    }
  }

  auto const& sipixelLinkColl = iEvent.get(sipixelSimLinksToken_);

  auto sipixelLinkVector = pruneByTpAssoc(sipixelLinkColl, selectedIds);

  auto sipixelOut = std::make_unique<edm::DetSetVector<PixelDigiSimLink>>(sipixelLinkVector);
  iEvent.put(std::move(sipixelOut), "siPixel");

  if (not sistripSimLinksToken_.isUninitialized()) {
    auto const& sistripLinkColl = iEvent.get(sistripSimLinksToken_);

    auto sistripLinkVector = pruneByTpAssoc(sistripLinkColl, selectedIds);

    auto sistripOut = std::make_unique<edm::DetSetVector<StripDigiSimLink>>(sistripLinkVector);
    iEvent.put(std::move(sistripOut), "siStrip");
  }

  if (not siphase2OTSimLinksToken_.isUninitialized()) {
    auto const& siphase2OTLinkColl = iEvent.get(siphase2OTSimLinksToken_);

    auto siphase2OTLinkVector = pruneByTpAssoc(siphase2OTLinkColl, selectedIds);

    auto siphase2OTOut = std::make_unique<edm::DetSetVector<PixelDigiSimLink>>(siphase2OTLinkVector);
    iEvent.put(std::move(siphase2OTOut), "siphase2OT");
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DigiSimLinkPruner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackingParticles");
  desc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis"));
  desc.addOptional<edm::InputTag>("stripSimLinkSrc");
  desc.addOptional<edm::InputTag>("phase2OTSimLinkSrc");

  descriptions.add("digiSimLinkPrunerDefault", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DigiSimLinkPruner);
