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
// $Date: 2006/09/20 21:35:41 $
// $Revision: 1.12 $
//

#include <memory>
#include <string>

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMaker.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace cms
{

  RoadSearchCloudMaker::RoadSearchCloudMaker(edm::ParameterSet const& conf) : 
    roadSearchCloudMakerAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<RoadSearchCloudCollection>();
  }


  // Virtual destructor needed.
  RoadSearchCloudMaker::~RoadSearchCloudMaker() { }  

  // Functions that gets called by framework every event
  void RoadSearchCloudMaker::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input SeedCollection
    std::string seedProducer = conf_.getParameter<std::string>("SeedProducer");

    // Step A: Get Inputs 
    edm::Handle<TrajectorySeedCollection> seeds;
    e.getByLabel(seedProducer, seeds);
    
    // retrieve InputTags for strip rechits
    edm::InputTag matchedStripRecHitsInputTag = conf_.getParameter<edm::InputTag>("matchedStripRecHits");
    edm::InputTag rphiStripRecHitsInputTag    = conf_.getParameter<edm::InputTag>("rphiStripRecHits");
    edm::InputTag stereoStripRecHitsInputTag  = conf_.getParameter<edm::InputTag>("stereoStripRecHits");
    
    // get Inputs
    edm::Handle<SiStripMatchedRecHit2DCollection> matchedRecHits;
    e.getByLabel(matchedStripRecHitsInputTag ,matchedRecHits);
    edm::Handle<SiStripRecHit2DCollection> rphiRecHits;
    e.getByLabel(rphiStripRecHitsInputTag ,rphiRecHits);
    edm::Handle<SiStripRecHit2DCollection> stereoRecHits;
    e.getByLabel(stereoStripRecHitsInputTag ,stereoRecHits);
    
    // retrieve InputTag for pixel rechits
    edm::InputTag pixelRecHitsInputTag  = conf_.getParameter<edm::InputTag>("pixelRecHits");
    
    // special treatment for getting pixel collection
    // if collection exists in file, use collection from file
    // if collection does not exist in file, create empty collection
    const SiPixelRecHitCollection *pixelRecHitCollection = 0;
    try {
      edm::Handle<SiPixelRecHitCollection> pixelRecHits;
      e.getByLabel(pixelRecHitsInputTag, pixelRecHits);
      pixelRecHitCollection = pixelRecHits.product();
    }
    catch (edm::Exception const& x) {
      if ( x.categoryCode() == edm::errors::ProductNotFound ) {
	if ( x.history().size() == 1 ) {
	  static const SiPixelRecHitCollection s_empty;
	  pixelRecHitCollection = &s_empty;
	  edm::LogWarning("RoadSearch") << "Collection SiPixelRecHitCollection with InputTag " << pixelRecHitsInputTag << " cannot be found, using empty collection of same type. The RoadSearch algorithm is also fully functional without Pixel RecHits.";
	}
      }
    }

    // Step B: create empty output collection
    std::auto_ptr<RoadSearchCloudCollection> output(new RoadSearchCloudCollection);

    // Step C: Invoke the seed finding algorithm
    roadSearchCloudMakerAlgorithm_.run(seeds,
				       rphiRecHits.product(),  
				       stereoRecHits.product(),
				       matchedRecHits.product(),
				       pixelRecHitCollection,
				       es,
				       *output);

    // Step D: write output to file
    e.put(output);

  }

}
