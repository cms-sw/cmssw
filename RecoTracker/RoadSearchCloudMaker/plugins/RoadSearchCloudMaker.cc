//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudMaker
// 
// Description:     Calls RoadSeachCloudMakerAlgorithm
//                  to find RoadSearchClouds.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: noeding $
// $Date: 2008/04/15 13:21:26 $
// $Revision: 1.20 $
//

#include <memory>
#include <string>

#include "RoadSearchCloudMaker.h"

#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

RoadSearchCloudMaker::RoadSearchCloudMaker(edm::ParameterSet const& conf) : 
  roadSearchCloudMakerAlgorithm_(conf) ,
  conf_(conf)
{
  produces<RoadSearchCloudCollection>();

  // retrieve InputTags for rechits
  matchedStripRecHitsInputTag_ = conf_.getParameter<edm::InputTag>("matchedStripRecHits");
  rphiStripRecHitsInputTag_    = conf_.getParameter<edm::InputTag>("rphiStripRecHits");
  stereoStripRecHitsInputTag_  = conf_.getParameter<edm::InputTag>("stereoStripRecHits");
  pixelRecHitsInputTag_        = conf_.getParameter<edm::InputTag>("pixelRecHits");
    
  // retrieve InputTags of input SeedCollection
  seedProducer_                = conf_.getParameter<edm::InputTag>("SeedProducer");

}


// Virtual destructor needed.
RoadSearchCloudMaker::~RoadSearchCloudMaker() { }  

// Functions that gets called by framework every event
void RoadSearchCloudMaker::produce(edm::Event& e, const edm::EventSetup& es)
{

  // Step A: Get Inputs 
  edm::Handle<RoadSearchSeedCollection> seedHandle;
  e.getByLabel(seedProducer_, seedHandle);
    
  // get Inputs
  edm::Handle<SiStripRecHit2DCollection> rphirecHitHandle;
  e.getByLabel(rphiStripRecHitsInputTag_ ,rphirecHitHandle);
  const SiStripRecHit2DCollection *rphiRecHitCollection = rphirecHitHandle.product();
  edm::Handle<SiStripRecHit2DCollection> stereorecHitHandle;
  e.getByLabel(stereoStripRecHitsInputTag_ ,stereorecHitHandle);
  const SiStripRecHit2DCollection *stereoRecHitCollection = stereorecHitHandle.product();

  // special treatment for getting matched RecHit collection
  // if collection exists in file, use collection from file
  // if collection does not exist in file, create empty collection
  static const SiStripMatchedRecHit2DCollection s_empty0;
  const SiStripMatchedRecHit2DCollection *matchedRecHitCollection = &s_empty0;
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedRecHits;
  if( e.getByLabel(matchedStripRecHitsInputTag_, matchedRecHits)) {
    matchedRecHitCollection = matchedRecHits.product();
  } else {
    LogDebug("RoadSearch") << "Collection SiStripMatchedRecHit2DCollection with InputTag " << matchedStripRecHitsInputTag_ << " cannot be found, using empty collection of same type. The RoadSearch algorithm is also fully functional without matched RecHits.";
  }
  
  // special treatment for getting pixel collection
  // if collection exists in file, use collection from file
  // if collection does not exist in file, create empty collection
  static const SiPixelRecHitCollection s_empty1;
  const SiPixelRecHitCollection *pixelRecHitCollection = &s_empty1;
  edm::Handle<SiPixelRecHitCollection> pixelRecHits;
  if( e.getByLabel(pixelRecHitsInputTag_, pixelRecHits)) {
    pixelRecHitCollection = pixelRecHits.product();
  } else {
    LogDebug("RoadSearch") << "Collection SiPixelRecHitCollection with InputTag " << pixelRecHitsInputTag_ << " cannot be found, using empty collection of same type. The RoadSearch algorithm is also fully functional without Pixel RecHits.";
  }
  

  // Step B: create empty output collection
  std::auto_ptr<RoadSearchCloudCollection> output(new RoadSearchCloudCollection);

  // Step C: Invoke the seed finding algorithm
  roadSearchCloudMakerAlgorithm_.run(seedHandle,
				     rphiRecHitCollection,  
				     stereoRecHitCollection,
				     matchedRecHitCollection,
				     pixelRecHitCollection,
				     es,
				     *output);

  // Step D: write output to file
  e.put(output);

}
