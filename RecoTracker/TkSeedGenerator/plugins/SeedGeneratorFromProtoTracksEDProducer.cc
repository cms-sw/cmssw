#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromProtoTrack.h"

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

    SeedFromProtoTrack seedFromProtoTrack( proto, es);
    if (seedFromProtoTrack.isValid()) (*result).push_back( seedFromProtoTrack.trajectorySeed() );
  }
  ev.put(result);
}




