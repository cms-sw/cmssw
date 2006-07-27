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
// $Author: tboccali $
// $Date: 2006/07/26 13:16:44 $
// $Revision: 1.7 $
//

#include <memory>
#include <string>

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMaker.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

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

    // retrieve producer name of input SiStripRecHit2DCollection
    std::string recHitProducer = conf_.getParameter<std::string>("RecHitProducer");
     // retrieve producer name of input SiPixelRecHitCollection - TMoulik
    std::string recHitProducer1 = conf_.getParameter<std::string>("RecHitProducer1");
 
    // get Inputs 
    edm::Handle<SiStripRecHit2DCollection> rphirecHits;
    e.getByLabel(recHitProducer, "rphiRecHit", rphirecHits);
    edm::Handle<SiStripRecHit2DCollection> stereorecHits;
    e.getByLabel(recHitProducer, "stereoRecHit", stereorecHits);

    edm::Handle<SiPixelRecHitCollection> pixRecHits; // TMoulik
    std::string recHitCollLabel = conf_.getUntrackedParameter<std::string>("RecHitCollLabel","pixRecHitConverter");
    e.getByLabel(recHitProducer1,pixRecHits); // TMoulik

    // Step B: create empty output collection
    std::auto_ptr<RoadSearchCloudCollection> output(new RoadSearchCloudCollection);

    // Step C: Invoke the seed finding algorithm
    roadSearchCloudMakerAlgorithm_.run(seeds,rphirecHits.product(),
				       stereorecHits.product(),pixRecHits.product(),
				       es,*output);

    // Step D: write output to file
    e.put(output);

  }

}
