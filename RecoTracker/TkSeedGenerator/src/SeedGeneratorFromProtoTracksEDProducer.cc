#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

SeedGeneratorFromProtoTracksEDProducer::SeedGeneratorFromProtoTracksEDProducer(const ParameterSet& cfg)
  : theInputCollectionTag(cfg.getParameter<InputTag>("InputCollection"))
{
  produces<TrajectorySeedCollection>();
}

void SeedGeneratorFromProtoTracksEDProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
  Handle<reco::TrackCollection> trks;
  ev.getByLabel(theInputCollectionTag, trks);
  ev.put(result);
}




