#include "SeedCombiner.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

using namespace edm;
//using namespace reco;

SeedCombiner::SeedCombiner(
    const edm::ParameterSet& cfg) 
  : 
  seedPairCollectionName_(cfg.getParameter<InputTag>("PairCollection")),
  seedTripletCollectionName_(cfg.getParameter<InputTag>("TripletCollection"))
{
    produces<TrajectorySeedCollection>();
}


SeedCombiner::~SeedCombiner()
{
}


void SeedCombiner::beginJob(const edm::EventSetup& es)
{
}

void SeedCombiner::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());

   Handle<TrajectorySeedCollection > seedPairList;
   Handle<TrajectorySeedCollection > seedTripletList;

   ev.getByLabel(seedPairCollectionName_,seedPairList);
   ev.getByLabel(seedTripletCollectionName_,seedTripletList);

   //std::cout << "=== collection triplets: " << seedTripletList->size() << std::endl;
   //std::cout << "=== collection pairs: " << seedPairList->size() << std::endl;

   result->reserve( seedTripletList->size() + seedPairList->size() );
   result->insert(result->end(), seedTripletList->begin(), seedTripletList->end() );
   result->insert(result->end(), seedPairList->begin()   , seedPairList->end()    );

   ev.put(result);
}
