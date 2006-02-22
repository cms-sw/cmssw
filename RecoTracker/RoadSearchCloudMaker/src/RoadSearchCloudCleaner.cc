//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudCleaner
// 
// Description:     Calls RoadSeachCloudCleanerAlgorithm
//                  to find RoadSearchClouds.
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/02/10 22:54:52 $
// $Revision: 1.3 $
//

#include <memory>
#include <string>

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudCleaner.h"

#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{

  RoadSearchCloudCleaner::RoadSearchCloudCleaner(edm::ParameterSet const& conf) : 
    roadSearchCloudCleanerAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<RoadSearchCloudCollection>();
  }


  // Virtual destructor needed.
  RoadSearchCloudCleaner::~RoadSearchCloudCleaner() { }  

  // Functions that gets called by framework every event
  void RoadSearchCloudCleaner::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // Step A: Get Inputs 

    // retrieve producer name of raw CloudCollection
    std::string rawcloudProducer = conf_.getParameter<std::string>("RawCloudProducer");
    edm::Handle<RoadSearchCloudCollection> rawclouds;
    e.getByLabel(rawcloudProducer, rawclouds);

    // retrieve producer name of input SeedCollection
    std::string seedProducer = conf_.getParameter<std::string>("SeedProducer");
    edm::Handle<TrackingSeedCollection> seeds;
    e.getByLabel(seedProducer, seeds);

    // Step B: create empty output collection
    std::auto_ptr<RoadSearchCloudCollection> output(new RoadSearchCloudCollection);

    // Step C: Invoke the cloud cleaning algorithm
    roadSearchCloudCleanerAlgorithm_.run(rawclouds.product(),seeds.product(),es,*output);

    // Step D: write output to file
    e.put(output);

  }

}
