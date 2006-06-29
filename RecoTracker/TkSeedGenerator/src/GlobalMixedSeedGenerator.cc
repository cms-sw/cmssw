//
// Package:         RecoTracker/TkSeedGenerator
// Class:           GlobalMixedSeedGenerator
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/GlobalMixedSeedGenerator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

using namespace std;
GlobalMixedSeedGenerator::GlobalMixedSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf) ,combinatorialSeedGenerator(conf)
 {
  edm::LogInfo ("GlobalMixedSeedGenerator")<<"Enter the GlobalMixedSeedGenerator";
  produces<TrajectorySeedCollection>();
}


// Virtual destructor needed.
GlobalMixedSeedGenerator::~GlobalMixedSeedGenerator() { }  

// Functions that gets called by framework every event
void GlobalMixedSeedGenerator::produce(edm::Event& e, const edm::EventSetup& es)
{  
  // get Inputs
  std::string pxlHitProducer = conf_.getParameter<std::string>("PixelHitProducer");
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  e.getByLabel(pxlHitProducer, pixelHits);

  std::string stripHitProducer = conf_.getParameter<std::string>("StripHitProducer");
  edm::Handle<SiStripRecHit2DMatchedLocalPosCollection> matchedrecHits;
  e.getByLabel(stripHitProducer,"matchedRecHit" ,matchedrecHits);
  edm::Handle<SiStripRecHit2DLocalPosCollection> rphirecHits;
  e.getByLabel(stripHitProducer,"rphiRecHit" ,rphirecHits);
  edm::Handle<SiStripRecHit2DLocalPosCollection> stereorecHits;
  e.getByLabel(stripHitProducer,"stereoRecHit" ,stereorecHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  //

  combinatorialSeedGenerator.init(*pixelHits,*matchedrecHits,*stereorecHits,*rphirecHits,es);
  combinatorialSeedGenerator.run(*output,es);

  // write output to file
  LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();


  e.put(output);
}
