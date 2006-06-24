//
// Package:         RecoTracker/TkSeedGenerator
// Class:           GlobalPixelLessSeedGenerator
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelLessSeedGenerator.h"
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
GlobalPixelLessSeedGenerator::GlobalPixelLessSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf) ,combinatorialSeedGenerator(conf)
 {
  edm::LogInfo ("GlobalPixelLessSeedGenerator")<<"Enter the GlobalPixelLessSeedGenerator";
  produces<TrajectorySeedCollection>();
}


// Virtual destructor needed.
GlobalPixelLessSeedGenerator::~GlobalPixelLessSeedGenerator() { }  

// Functions that gets called by framework every event
void GlobalPixelLessSeedGenerator::produce(edm::Event& e, const edm::EventSetup& es)
{  
  // get Inputs
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  edm::Handle<SiStripRecHit2DMatchedLocalPosCollection> matchedrecHits;
  e.getByLabel(hitProducer,"matchedRecHit" ,matchedrecHits);
  edm::Handle<SiStripRecHit2DLocalPosCollection> rphirecHits;
  e.getByLabel(hitProducer,"rphiRecHit" ,rphirecHits);
  edm::Handle<SiStripRecHit2DLocalPosCollection> stereorecHits;
  e.getByLabel(hitProducer,"stereoRecHit" ,stereorecHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  //

  combinatorialSeedGenerator.init(*matchedrecHits,*stereorecHits,*rphirecHits,es);
  combinatorialSeedGenerator.run(*output,es);

  // write output to file
  LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();


  e.put(output);
}
