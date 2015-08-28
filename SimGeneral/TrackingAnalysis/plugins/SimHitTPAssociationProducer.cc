#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <utility>

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"

#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"

SimHitTPAssociationProducer::SimHitTPAssociationProducer(const edm::ParameterSet & cfg) 
  : _simHitSrc(),
    _trackingParticleSrc(consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticleSrc")))
{
  produces<SimHitTPAssociationList>();
  std::vector<edm::InputTag> tags = cfg.getParameter<std::vector<edm::InputTag> >("simHitSrc");
  _simHitSrc.reserve(tags.size());
  for(auto const& tag  : tags) {
    _simHitSrc.emplace_back(consumes<edm::PSimHitContainer>(tag));
  }
}

SimHitTPAssociationProducer::~SimHitTPAssociationProducer() {
}
		
void SimHitTPAssociationProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  std::auto_ptr<SimHitTPAssociationList> simHitTPList(new SimHitTPAssociationList);
 
  // TrackingParticle
  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  iEvent.getByToken(_trackingParticleSrc,  TPCollectionH);

  // prepare temporary map between SimTrackId and TrackingParticle index
  std::map<std::pair<size_t, EncodedEventId>, TrackingParticleRef> mapping;
  for (TrackingParticleCollection::size_type itp = 0; itp < TPCollectionH.product()->size(); ++itp) {
    TrackingParticleRef trackingParticle(TPCollectionH, itp);
    // SimTracks inside TrackingParticle
    EncodedEventId eid(trackingParticle->eventId());
    for (auto itrk  = trackingParticle->g4Track_begin(); itrk != trackingParticle->g4Track_end(); ++itrk) {
      std::pair<uint32_t, EncodedEventId> trkid(itrk->trackId(), eid);
      mapping.insert(std::make_pair(trkid, trackingParticle));
    }
  }

  // PSimHits
  for (auto const& psit : _simHitSrc ) {
    edm::Handle<edm::PSimHitContainer>  PSimHitCollectionH;
    iEvent.getByToken(psit,  PSimHitCollectionH);
    for (unsigned int psimHit = 0;psimHit != PSimHitCollectionH->size();++psimHit) {
      TrackPSimHitRef pSimHitRef(PSimHitCollectionH,psimHit);
      std::pair<uint32_t, EncodedEventId> simTkIds(pSimHitRef->trackId(),pSimHitRef->eventId()); 
      auto ipos = mapping.find(simTkIds);
      if (ipos != mapping.end()) {
	simHitTPList->push_back(std::make_pair(ipos->second,pSimHitRef));
      }
    }
  } 
  
  std::sort(simHitTPList->begin(),simHitTPList->end(),simHitTPAssociationListGreater);
  iEvent.put(simHitTPList);

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SimHitTPAssociationProducer);
