// -*- C++ -*-
//
// Package:    SimTracker/SimLinkPruner
// Class:      SimLinkPruner
//
/**\class SimLinkPruner SimLinkPruner.cc SimTracker/SimLinkPruner/plugins/SimLinkPruner.cc

 Description: [one line class summary]

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
#include "FWCore/Framework/interface/stream/EDProducer.h"

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

class SimLinkPruner : public edm::stream::EDProducer<> {
public:
  explicit SimLinkPruner(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> sipixelSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink>> sistripSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> siphase2OTSimLinksToken_;
};

SimLinkPruner::SimLinkPruner(const edm::ParameterSet& iConfig)
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
void SimLinkPruner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<TrackingParticleCollection> TPCollectionH;

  iEvent.getByToken(trackingParticleToken_, TPCollectionH);

  auto const& tpColl = *TPCollectionH.product();

  std::unordered_set<UniqueSimTrackId, UniqueSimTrackIdHash> selectedIds;
  for (TrackingParticleCollection::size_type itp = 0; itp < tpColl.size(); ++itp) {
    TrackingParticleRef trackingParticleRef(TPCollectionH, itp);

    auto const& trackingParticle = tpColl[itp];

    // SimTracks inside TrackingParticle

    EncodedEventId eid(trackingParticle.eventId());

    for (auto const& trk : trackingParticle.g4Tracks()) {
      selectedIds.emplace(trk.trackId(), eid);
    }
  }

  edm::Handle<edm::DetSetVector<PixelDigiSimLink>> siPixelSimLinksH;

  iEvent.getByToken(sipixelSimLinksToken_, siPixelSimLinksH);

  auto const& sipixelLinkColl = *siPixelSimLinksH.product();

  auto sipixelLinkVector = pruneByTpAssoc(sipixelLinkColl, selectedIds);

  auto sipixelOut = std::make_unique<edm::DetSetVector<PixelDigiSimLink>>(sipixelLinkVector);
  iEvent.put(std::move(sipixelOut), "siPixel");

  if (not sistripSimLinksToken_.isUninitialized()) {
    edm::Handle<edm::DetSetVector<StripDigiSimLink>> siStripSimLinksH;

    iEvent.getByToken(sistripSimLinksToken_, siStripSimLinksH);

    auto const& sistripLinkColl = *siStripSimLinksH.product();

    auto sistripLinkVector = pruneByTpAssoc(sistripLinkColl, selectedIds);

    auto sistripOut = std::make_unique<edm::DetSetVector<StripDigiSimLink>>(sistripLinkVector);
    iEvent.put(std::move(sistripOut), "siStrip");
  }

  if (not siphase2OTSimLinksToken_.isUninitialized()) {
    edm::Handle<edm::DetSetVector<PixelDigiSimLink>> siphase2OTSimLinksH;

    iEvent.getByToken(siphase2OTSimLinksToken_, siphase2OTSimLinksH);

    auto const& siphase2OTLinkColl = *siphase2OTSimLinksH.product();

    auto siphase2OTLinkVector = pruneByTpAssoc(siphase2OTLinkColl, selectedIds);

    auto siphase2OTOut = std::make_unique<edm::DetSetVector<PixelDigiSimLink>>(siphase2OTLinkVector);
    iEvent.put(std::move(siphase2OTOut), "siphase2OT");
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SimLinkPruner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis"));
  desc.addOptional<edm::InputTag>("stripSimLinkSrc");
  desc.addOptional<edm::InputTag>("phase2OTSimLinkSrc");

  descriptions.add("pruneSimLinkDefault", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimLinkPruner);
