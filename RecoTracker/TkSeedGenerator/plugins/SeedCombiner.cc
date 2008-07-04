#include "SeedCombiner.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

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
  SeedPairCollectionName_(cfg.getUntrackedParameter<std::string>("PairCollection")),
  SeedTripletCollectionName_(cfg.getUntrackedParameter<std::string>("TripletCollection"))
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

   Handle<TrajectorySeedCollection > SeedPairList;
   Handle<TrajectorySeedCollection > SeedTripletList;

   ev.getByLabel(SeedPairCollectionName_,SeedPairList);
   ev.getByLabel(SeedTripletCollectionName_,SeedTripletList);

   //std::cout << "=== collection triplets: " << SeedTripletList->size() << std::endl;
   //std::cout << "=== collection pairs: " << SeedPairList->size() << std::endl;

   result->reserve( SeedTripletList->size() + SeedPairList->size() );
   result->insert(result->end(), SeedTripletList->begin(), SeedTripletList->end() );
   result->insert(result->end(), SeedPairList->begin()   , SeedPairList->end()    );

   ev.put(result);
}
