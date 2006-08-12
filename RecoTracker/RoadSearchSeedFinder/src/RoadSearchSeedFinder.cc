//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchSeedFinder
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrajectorySeeds.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: tboccali $
// $Date: 2006/07/24 19:44:42 $
// $Revision: 1.6 $
//

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinder.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

RoadSearchSeedFinder::RoadSearchSeedFinder(edm::ParameterSet const& conf) : 
  roadSearchSeedFinderAlgorithm_(conf) ,
  conf_(conf)
{
  produces<TrajectorySeedCollection>();

}


// Virtual destructor needed.
RoadSearchSeedFinder::~RoadSearchSeedFinder() { }  

// Functions that gets called by framework every event
void RoadSearchSeedFinder::produce(edm::Event& e, const edm::EventSetup& es)
{

  // retrieve producer name of input SiStripRecHit2DCollection
  std::string recHitProducer = conf_.getParameter<std::string>("RecHitProducer");
  
  // get Inputs
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedRecHits;
  e.getByLabel(recHitProducer,"matchedRecHit" ,matchedRecHits);
  edm::Handle<SiStripRecHit2DCollection> rphiRecHits;
  e.getByLabel(recHitProducer,"rphiRecHit" ,rphiRecHits);
  edm::Handle<SiStripRecHit2DCollection> stereoRecHits;
  e.getByLabel(recHitProducer,"stereoRecHit" ,stereoRecHits);
 
  // special treatment for getting pixel collection
  // if collection exists in file, use collection from file
  // if collection does not exist in file, create empty collection
  const SiPixelRecHitCollection *pixelRecHitCollection = 0;
  
  try {
    edm::Handle<SiPixelRecHitCollection> pixelRecHits;
    //e.getByLabel(pixelRecHitProducer, pixelRecHits);
    e.getByLabel(recHitProducer, pixelRecHits);
    pixelRecHitCollection = pixelRecHits.product();
  }
  catch (edm::Exception const& x) {
    if ( x.categoryCode() == edm::errors::ProductNotFound ) {
      if ( x.history().size() == 1 ) {
	pixelRecHitCollection = new SiPixelRecHitCollection();
      }
    }
  }
  
  // create empty output collection
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  
  // invoke the seed finding algorithm
  roadSearchSeedFinderAlgorithm_.run(rphiRecHits.product(),  
				     stereoRecHits.product(),
				     matchedRecHits.product(),
				     pixelRecHitCollection,
				     es,
				     *output);
  
  // write output to file
  e.put(output);

}
