#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"

#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"

namespace {
  using TkId = std::pair<uint32_t, EncodedEventId>;
  struct TkIdHash
  {
    std::size_t operator()(TkId const& s) const noexcept
    {
          std::size_t h1 = std::hash<uint32_t>{}(s.first);
           std::size_t h2 = std::hash<uint32_t>{}(s.second.rawId());
           return h1 ^ (h2 << 1);
    }
};
}


SimHitTPAssociationProducer::SimHitTPAssociationProducer(const edm::ParameterSet &cfg)
    : _simHitSrc(),
      _trackingParticleSrc(
          consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticleSrc"))) {
  produces<SimHitTPAssociationList>();
  std::vector<edm::InputTag> tags = cfg.getParameter<std::vector<edm::InputTag>>("simHitSrc");
  _simHitSrc.reserve(tags.size());
  for (auto const &tag : tags) {
    _simHitSrc.emplace_back(consumes<edm::PSimHitContainer>(tag));
  }
}

SimHitTPAssociationProducer::~SimHitTPAssociationProducer() {}

void SimHitTPAssociationProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &es) const {
  std::unique_ptr<SimHitTPAssociationList> simHitTPList(new SimHitTPAssociationList);

  // TrackingParticle
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(_trackingParticleSrc, TPCollectionH);

  // prepare temporary map between SimTrackId and TrackingParticle index
  std::unordered_map<TkId, TrackingParticleRef, TkIdHash> mapping;
  auto const & tpColl = *TPCollectionH.product();
  for (TrackingParticleCollection::size_type itp = 0, size =tpColl.size(); itp < size; ++itp) {
    auto const & trackingParticle = tpColl[itp];
    TrackingParticleRef trackingParticleRef(TPCollectionH, itp);
    // SimTracks inside TrackingParticle
    EncodedEventId eid(trackingParticle.eventId());
    for (auto const & trk : trackingParticle.g4Tracks()) {
      TkId trkid(trk.trackId(), eid);
      mapping.insert(std::make_pair(trkid, trackingParticleRef));
    }
  }

  // PSimHits
  for (auto const &psit : _simHitSrc) {
    edm::Handle<edm::PSimHitContainer> PSimHitCollectionH;
    iEvent.getByToken(psit, PSimHitCollectionH);
    auto const & pSimHitCollection = *PSimHitCollectionH;
    for (unsigned int psimHitI = 0, size=pSimHitCollection.size(); psimHitI < size; ++psimHitI) {
      TrackPSimHitRef pSimHitRef(PSimHitCollectionH, psimHitI);
      auto const & pSimHit = pSimHitCollection[psimHitI];
      TkId simTkIds(pSimHit.trackId(), pSimHit.eventId());
      auto ipos = mapping.find(simTkIds);
      if (ipos != mapping.end()) {
        simHitTPList->push_back(std::make_pair(ipos->second, pSimHitRef));
      }
    }
  }

  std::sort(simHitTPList->begin(), simHitTPList->end(), simHitTPAssociationListGreater);
  iEvent.put(std::move(simHitTPList));
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_MODULE(SimHitTPAssociationProducer);
