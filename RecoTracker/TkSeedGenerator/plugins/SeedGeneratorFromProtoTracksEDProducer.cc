#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromProtoTrack.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "SeedFromConsecutiveHitsCreator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using namespace edm;
using namespace reco;

template <class T> T sqr( T t) {return t*t;}
typedef TransientTrackingRecHit::ConstRecHitPointer Hit;

struct HitLessByRadius { bool operator() (const Hit& h1, const Hit & h2) { return h1->globalPosition().perp2() < h2->globalPosition().perp2(); } };

SeedGeneratorFromProtoTracksEDProducer::SeedGeneratorFromProtoTracksEDProducer(const ParameterSet& cfg)
  : theConfig(cfg), theInputCollectionTag(cfg.getParameter<InputTag>("InputCollection"))
{
  produces<TrajectorySeedCollection>();
}

void SeedGeneratorFromProtoTracksEDProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
  Handle<reco::TrackCollection> trks;
  ev.getByLabel(theInputCollectionTag, trks);

  const TrackCollection &protos = *(trks.product());
  
  for (TrackCollection::const_iterator it=protos.begin(); it!= protos.end(); ++it) {
    const Track & proto = (*it);

    if (theConfig.getParameter<bool>("useProtoTrackKinematics")) {
      SeedFromProtoTrack seedFromProtoTrack( proto, es);
      if (seedFromProtoTrack.isValid()) (*result).push_back( seedFromProtoTrack.trajectorySeed() );
    } else {
      edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
      std::string builderName = theConfig.getParameter<std::string>("TTRHBuilder");
      es.get<TransientRecHitRecord>().get(builderName,ttrhbESH);
      std::vector<Hit> hits;
      for (unsigned int iHit = 0, nHits = proto.recHitsSize(); iHit < nHits; ++iHit) {
        TrackingRecHitRef refHit = proto.recHit(iHit);
        if(refHit->isValid()) hits.push_back(ttrhbESH->build(  &(*refHit) ));
        sort(hits.begin(), hits.end(), HitLessByRadius());
      }
      if (hits.size() >= 2) {
        GlobalPoint vtx(proto.vertex().x(), proto.vertex().y(), proto.vertex().z());
        double mom_perp = sqrt(proto.momentum().x()*proto.momentum().x()+proto.momentum().y()*proto.momentum().y());
        GlobalTrackingRegion region(mom_perp, vtx, 0.2, 0.2);
        SeedFromConsecutiveHitsCreator().trajectorySeed(*result, SeedingHitSet(hits), region, es);
      }
    }
  } 

  ev.put(result);
}

