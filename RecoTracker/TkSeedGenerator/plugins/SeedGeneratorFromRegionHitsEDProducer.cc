#include "SeedGeneratorFromRegionHitsEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

SeedGeneratorFromRegionHitsEDProducer::SeedGeneratorFromRegionHitsEDProducer(const edm::ParameterSet& cfg)
    : theRegionProducer(nullptr),
      theClusterCheck(cfg.getParameter<edm::ParameterSet>("ClusterCheckPSet"), consumesCollector()) {
  theSilentOnClusterCheck =
      cfg.getParameter<edm::ParameterSet>("ClusterCheckPSet").getUntrackedParameter<bool>("silentClusterCheck", false);

  moduleName = cfg.getParameter<std::string>("@module_label");

  edm::ParameterSet creatorPSet = cfg.getParameter<edm::ParameterSet>("SeedCreatorPSet");

  edm::ParameterSet regfactoryPSet = cfg.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName, regfactoryPSet, consumesCollector());

  edm::ConsumesCollector iC = consumesCollector();
  edm::ParameterSet hitsfactoryPSet = cfg.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");

  edm::ParameterSet comparitorPSet = cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  std::unique_ptr<SeedComparitor> aComparitor;
  if (comparitorName != "none") {
    aComparitor = SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC);
  }

  std::string creatorName = creatorPSet.getParameter<std::string>("ComponentName");

  theGenerator = std::make_unique<SeedGeneratorFromRegionHits>(
      OrderedHitsGeneratorFactory::get()->create(hitsfactoryName, hitsfactoryPSet, iC),
      std::move(aComparitor),
      SeedCreatorFactory::get()->create(creatorName, creatorPSet));

  produces<TrajectorySeedCollection>();
}

SeedGeneratorFromRegionHitsEDProducer::~SeedGeneratorFromRegionHitsEDProducer() {}

void SeedGeneratorFromRegionHitsEDProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  auto triplets = std::make_unique<TrajectorySeedCollection>();

  //protection for big ass events...
  size_t clustsOrZero = theClusterCheck.tooManyClusters(ev);
  if (clustsOrZero) {
    if (!theSilentOnClusterCheck)
      edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";
    ev.put(std::move(triplets));
    return;
  }

  typedef std::vector<std::unique_ptr<TrackingRegion> > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev, es);

  for (IR ir = regions.begin(), irEnd = regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion& region = **ir;

    // make job
    theGenerator->run(*triplets, region, ev, es);
    // std::cout << "created seeds for " << moduleName << " " << triplets->size() << std::endl;
  }
  triplets->shrink_to_fit();

  // put to event
  ev.put(std::move(triplets));
}
