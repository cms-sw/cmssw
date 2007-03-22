#include "SeedGeneratorFromRegionHitsEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"


SeedGeneratorFromRegionHitsEDProducer::SeedGeneratorFromRegionHitsEDProducer(
    const edm::ParameterSet& cfg) 
  : theConfig(cfg), theGenerator(0)
{
    produces<TrajectorySeedCollection>();
}

SeedGeneratorFromRegionHitsEDProducer::~SeedGeneratorFromRegionHitsEDProducer()
{
  delete theGenerator;
}

void SeedGeneratorFromRegionHitsEDProducer::beginJob(const edm::EventSetup& es)
{
  edm::ParameterSet regfactoryPSet = 
      theConfig.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  TrackingRegionProducer*  regionProducer = 
        TrackingRegionProducerFactory::get()->create( regfactoryName, regfactoryPSet);

  edm::ParameterSet hitsfactoryPSet = 
      theConfig.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  hitsGenerator = 
        OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet);

  theGenerator = new SeedGeneratorFromRegionHits(regionProducer,hitsGenerator,
     theConfig); // config is passed temporary!!!!
  
}


void SeedGeneratorFromRegionHitsEDProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
  theGenerator->run(*result, ev,es);
  ev.put(result);
}
