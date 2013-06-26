#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
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
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <cassert>
using namespace edm;
using namespace reco;

template <class T> T sqr( T t) {return t*t;}
typedef TransientTrackingRecHit::ConstRecHitPointer Hit;

struct HitLessByRadius { bool operator() (const Hit& h1, const Hit & h2) { return h1->globalPosition().perp2() < h2->globalPosition().perp2(); } };

SeedGeneratorFromProtoTracksEDProducer::SeedGeneratorFromProtoTracksEDProducer(const ParameterSet& cfg)
  : theConfig(cfg), 
    theInputCollectionTag(cfg.getParameter<InputTag>("InputCollection")),
    theInputVertexCollectionTag(cfg.getParameter<InputTag>("InputVertexCollection")),
    originHalfLength(cfg.getParameter<double>("originHalfLength")),
    originRadius(cfg.getParameter<double>("originRadius")),
    useProtoTrackKinematics(cfg.getParameter<bool>("useProtoTrackKinematics")),
    useEventsWithNoVertex(cfg.getParameter<bool>("useEventsWithNoVertex")),
    builderName(cfg.getParameter<std::string>("TTRHBuilder"))

{
  produces<TrajectorySeedCollection>();
}

void SeedGeneratorFromProtoTracksEDProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
  Handle<reco::TrackCollection> trks;
  ev.getByLabel(theInputCollectionTag, trks);

  const TrackCollection &protos = *(trks.product());
  
  edm::Handle<reco::VertexCollection> vertices;
  bool foundVertices = ev.getByLabel(theInputVertexCollectionTag, vertices);
  //const reco::VertexCollection & vertices = *(h_vertices.product());

  ///
  /// need optimization: all es stuff should go out of the loop
  /// 
  for (TrackCollection::const_iterator it=protos.begin(); it!= protos.end(); ++it) {
    const Track & proto = (*it);
    GlobalPoint vtx(proto.vertex().x(), proto.vertex().y(), proto.vertex().z());

    // check the compatibility with a primary vertex
    bool keepTrack = false;
    if ( !foundVertices ) { 
	  if (useEventsWithNoVertex) keepTrack = true;
    } else { 
      for (reco::VertexCollection::const_iterator iv=vertices->begin(); iv!= vertices->end(); ++iv) {
        GlobalPoint aPV(iv->position().x(),iv->position().y(),iv->position().z());
	double distR2 = sqr(vtx.x()-aPV.x()) +sqr(vtx.y()-aPV.y());
	double distZ = fabs(vtx.z()-aPV.z());
	if ( distR2 < sqr(originRadius) && distZ < originHalfLength ) { 
	  keepTrack = true;
	  break;
        }
      }
    }
    if (!keepTrack) continue;

    if ( useProtoTrackKinematics ) {
      SeedFromProtoTrack seedFromProtoTrack( proto, es);
      if (seedFromProtoTrack.isValid()) (*result).push_back( seedFromProtoTrack.trajectorySeed() );
    } else {
      edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
      es.get<TransientRecHitRecord>().get(builderName,ttrhbESH);
      std::vector<Hit> hits;
      for (unsigned int iHit = 0, nHits = proto.recHitsSize(); iHit < nHits; ++iHit) {
        TrackingRecHitRef refHit = proto.recHit(iHit);
        if(refHit->isValid()) hits.push_back(ttrhbESH->build(  &(*refHit) ));
      }
      sort(hits.begin(), hits.end(), HitLessByRadius());
      assert(hits.size()<4);
      if (hits.size() > 1) {
        double mom_perp = sqrt(proto.momentum().x()*proto.momentum().x()+proto.momentum().y()*proto.momentum().y());
	GlobalTrackingRegion region(mom_perp, vtx, 0.2, 0.2);
	SeedFromConsecutiveHitsCreator seedCreator;
	seedCreator.init(region, es, 0);
	seedCreator.makeSeed(*result, SeedingHitSet(hits[0], hits[1], hits.size() >2 ? hits[2] : SeedingHitSet::nullPtr() ));
      }
    }
  } 

  ev.put(result);
}

