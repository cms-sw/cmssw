//
// Package:         RecoTracker/TkSeedGenerator
// Class:           GlobalPixelLessSeedGenerator
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/SpecialSeedGenerators/interface/CosmicSeedGenerator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

using namespace std;
CosmicSeedGenerator::CosmicSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf) ,cosmic_seed(conf)
 {
  edm::LogInfo ("CosmicSeedGenerator")<<"Enter the CosmicSeedGenerator";
  produces<TrajectorySeedCollection>();
}


// Virtual destructor needed.
CosmicSeedGenerator::~CosmicSeedGenerator() { }  

// Functions that gets called by framework every event
void CosmicSeedGenerator::produce(edm::Event& ev, const edm::EventSetup& es)
{
  // get Inputs
  edm::InputTag matchedrecHitsTag = conf_.getParameter<edm::InputTag>("matchedRecHits");
  edm::InputTag rphirecHitsTag = conf_.getParameter<edm::InputTag>("rphirecHits");
  edm::InputTag stereorecHitsTag = conf_.getParameter<edm::InputTag>("stereorecHits");

  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  ev.getByLabel( rphirecHitsTag, rphirecHits );
  edm::Handle<SiStripRecHit2DCollection> stereorecHits;
  ev.getByLabel( stereorecHitsTag ,stereorecHits );
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits; 	 
  ev.getByLabel( matchedrecHitsTag ,matchedrecHits );
 

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);

  //check on the number of clusters
  ClusterChecker check(conf_);
  size_t clustsOrZero = check.tooManyClusters(ev);
  if (!clustsOrZero){
    cosmic_seed.init(*stereorecHits,*rphirecHits,*matchedrecHits, es);
    
    // invoke the seed finding algorithm
    cosmic_seed.run(*output,es);
  } else edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";

  // write output to file
  LogDebug("CosmicSeedGenerator")<<" number of seeds = "<< output->size();


  ev.put(output);
}
