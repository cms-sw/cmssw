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
// $Author: gutsche $
// $Date: 2006/03/28 22:54:07 $
// $Revision: 1.5 $
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
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits;
  e.getByLabel(recHitProducer,"matchedRecHit" ,matchedrecHits);
  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  e.getByLabel(recHitProducer,"rphiRecHit" ,rphirecHits);

  // create empty output collection
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  
  // invoke the seed finding algorithm
  roadSearchSeedFinderAlgorithm_.run(matchedrecHits,rphirecHits,es,*output);
  
  // write output to file
  e.put(output);

}
