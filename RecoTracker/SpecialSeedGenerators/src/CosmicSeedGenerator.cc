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


using namespace std;
CosmicSeedGenerator::CosmicSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf) ,cosmic_seed(conf),
  check(conf,consumesCollector())

 {
  edm::LogInfo ("CosmicSeedGenerator")<<"Enter the CosmicSeedGenerator";
  // get Inputs
  matchedrecHitsToken_ = consumes<SiStripMatchedRecHit2DCollection>(
      conf_.getParameter<edm::InputTag>("matchedRecHits"));
  rphirecHitsToken_ = consumes<SiStripRecHit2DCollection>(
          conf_.getParameter<edm::InputTag>("rphirecHits"));
  stereorecHitsToken_ = consumes<SiStripRecHit2DCollection>(
          conf_.getParameter<edm::InputTag>("stereorecHits"));

  produces<TrajectorySeedCollection>();
}


// Virtual destructor needed.
CosmicSeedGenerator::~CosmicSeedGenerator() { }  

// Functions that gets called by framework every event
void CosmicSeedGenerator::produce(edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  ev.getByToken( rphirecHitsToken_, rphirecHits);
  edm::Handle<SiStripRecHit2DCollection> stereorecHits;
  ev.getByToken( stereorecHitsToken_,stereorecHits);
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits; 	 
  ev.getByToken( matchedrecHitsToken_, matchedrecHits);
 

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);

  //check on the number of clusters
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
