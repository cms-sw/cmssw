//
// Package:         RecoTracker/TkSeedGenerator
// Class:           TkSeedGenerator
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

  // retrieve producer name of input SiStripRecHit2DLocalPosCollection
  //  std::string recHitProducer = conf_.getParameter<std::string>("RecHitProducer");
  
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  e.getByLabel("LocalMeasurementConverter","SiPixelRecHit" ,pixelHits);



   // create empty output collection
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  
  //
  //  globalseed.init(es); 
   // invoke the seed finding algorithm
  //   globalseed.run();  
  // write output to file
   // e.put(output);

}
