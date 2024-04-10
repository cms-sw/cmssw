#include <algorithm>
#include <map>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/TrackingAnalysis/interface/UniqueSimTrackId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"

SimHitTPAssociationProducer::SimHitTPAssociationProducer(const edm::ParameterSet &cfg)
    : _simHitSrc(),
      _trackingParticleSrc(
          consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticleSrc"))) {
  produces<SimHitTPAssociationList>();
  produces<SimTrackToTPMap>("simTrackToTP");
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
  auto simTrackToTPMap = std::make_unique<SimTrackToTPMap>();
  auto const &tpColl = *TPCollectionH.product();
  for (TrackingParticleCollection::size_type itp = 0, size = tpColl.size(); itp < size; ++itp) {
    auto const &trackingParticle = tpColl[itp];
    TrackingParticleRef trackingParticleRef(TPCollectionH, itp);
    // SimTracks inside TrackingParticle
    EncodedEventId eid(trackingParticle.eventId());
    for (auto const &trk : trackingParticle.g4Tracks()) {
      UniqueSimTrackId trkid(trk.trackId(), eid);
      simTrackToTPMap->mapping.insert(std::make_pair(trkid, trackingParticleRef));
    }
  }

  // PSimHits
  for (auto const &psit : _simHitSrc) {
    edm::Handle<edm::PSimHitContainer> PSimHitCollectionH;
    iEvent.getByToken(psit, PSimHitCollectionH);
    auto const &pSimHitCollection = *PSimHitCollectionH;
    for (unsigned int psimHitI = 0, size = pSimHitCollection.size(); psimHitI < size; ++psimHitI) {
      TrackPSimHitRef pSimHitRef(PSimHitCollectionH, psimHitI);
      auto const &pSimHit = pSimHitCollection[psimHitI];
      UniqueSimTrackId simTkIds(pSimHit.trackId(), pSimHit.eventId());
      auto ipos = simTrackToTPMap->mapping.find(simTkIds);
      if (ipos != simTrackToTPMap->mapping.end()) {
        simHitTPList->emplace_back(ipos->second, pSimHitRef);
      }
    }
  }

  std::sort(simHitTPList->begin(), simHitTPList->end(), simHitTPAssociationListGreater);
  iEvent.put(std::move(simHitTPList));
  iEvent.put(std::move(simTrackToTPMap), "simTrackToTP");
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_MODULE(SimHitTPAssociationProducer);
