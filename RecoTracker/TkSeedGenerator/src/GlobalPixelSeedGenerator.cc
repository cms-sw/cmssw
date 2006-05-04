//
// Package:         RecoTracker/TkSeedGenerator
// Class:           GlobalPixelSeedGenerator
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelSeedGenerator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

using namespace std;
GlobalPixelSeedGenerator::GlobalPixelSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf) ,globalseed(conf)
 {
  edm::LogInfo ("GlobalPixelSeedGenerator")<<"Enter the GlobalPixelSeedGenerator";
  produces<TrajectorySeedCollection>();
}


// Virtual destructor needed.
GlobalPixelSeedGenerator::~GlobalPixelSeedGenerator() { }  

// Functions that gets called by framework every event
void GlobalPixelSeedGenerator::produce(edm::Event& e, const edm::EventSetup& es)
{


  
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);


  //
  // get the pixel Hits
  //
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  e.getByLabel(hitProducer, pixelHits);

  //  e.getByType(pixelHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  //

  globalseed.init(*pixelHits,es);

  // invoke the seed finding algorithm
  globalseed.run(*output,es);

  // write output to file
  LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();


  e.put(output);
}
