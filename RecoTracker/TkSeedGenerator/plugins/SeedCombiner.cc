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

#include "RecoTracker/TrackProducer/interface/ClusterRemovalRefSetter.h"

using namespace edm;
//using namespace reco;

SeedCombiner::SeedCombiner(
    const edm::ParameterSet& cfg) 
  : 
    inputCollections_(cfg.getParameter<std::vector<edm::InputTag> >("seedCollections"))
{
    produces<TrajectorySeedCollection>();
    reKeing_=false;
    if (cfg.exists("clusterRemovalInfos")){
      clusterRemovalInfos_=cfg.getParameter<std::vector<edm::InputTag> >("clusterRemovalInfos");
      if (clusterRemovalInfos_.size()!=0 && clusterRemovalInfos_.size()==inputCollections_.size()) reKeing_=true;
    }
}


SeedCombiner::~SeedCombiner()
{
}


void SeedCombiner::produce(edm::Event& ev, const edm::EventSetup& es)
{
    // Read inputs, and count total seeds
    size_t ninputs = inputCollections_.size();
    size_t nseeds = 0;
    std::vector<Handle<TrajectorySeedCollection > > seedCollections(ninputs);
    for (size_t i = 0; i < ninputs; ++i) {
        ev.getByLabel(inputCollections_[i], seedCollections[i]);
        nseeds += seedCollections[i]->size();
    }

    // Prepare output collections, with the correct capacity
    std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
    result->reserve( nseeds );

    // Write into output collection
    unsigned int iSC=0,iSC_max=seedCollections.size();
    for (;iSC!=iSC_max;++iSC){
      Handle<TrajectorySeedCollection> & collection=seedCollections[iSC];
      if (reKeing_ && !(clusterRemovalInfos_[iSC]==edm::InputTag(""))){
	ClusterRemovalRefSetter refSetter(ev, clusterRemovalInfos_[iSC]);
	
	for (TrajectorySeedCollection::const_iterator iS=collection->begin();
	     iS!=collection->end();++iS){
	  TrajectorySeed::recHitContainer  newRecHitContainer;
	  newRecHitContainer.reserve(iS->nHits());
	  TrajectorySeed::const_iterator iH=iS->recHits().first;
	  TrajectorySeed::const_iterator iH_end=iS->recHits().second;
	  //loop seed rechits, copy over and rekey.
	  for (;iH!=iH_end;++iH){
	    newRecHitContainer.push_back(*iH);
	    refSetter.reKey(&newRecHitContainer.back());
	  }
	  result->push_back(TrajectorySeed(iS->startingState(),
					   std::move(newRecHitContainer),
					   iS->direction()));
	}
      }else{
	//just insert the new seeds as they are
	result->insert(result->end(), collection->begin(), collection->end());
      }
    }

    // Save result into the event
    ev.put(result);
}
