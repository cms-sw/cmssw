#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using namespace edm;
using namespace reco;

template <class T> T sqr( T t) {return t*t;}

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

    //
    // temporary solution, reusing  SeedFromConsecutiveHits,
    // should be replaced by dedicated seed buildr 
    //
    std::vector<const TrackingRecHit* > recHits;
    for (unsigned int iHit = 0, nHits = proto.recHitsSize(); iHit < nHits; ++iHit) {  
      TrackingRecHitRef refHit = proto.recHit(iHit);
      recHits.push_back( &(*refHit) );
    }
    if (recHits.size() < 2) continue; 
    GlobalPoint vtx(0.,0.,0.);
    double originRBound = 0.2;
    double originZBound = 15.9;
    GlobalError vtxerr( sqr(originRBound), 0, sqr(originRBound),
                                        0, 0, sqr(originZBound) );
  
    SeedFromConsecutiveHits seedfromhits( recHits[1], recHits[0], vtx, vtxerr, es, theConfig);
    if (seedfromhits.isValid()) (*result).push_back( seedfromhits.TrajSeed() );
  }
  ev.put(result);
}




