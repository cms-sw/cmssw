//
// Package:         RecoTracker/RoadSearchCloudCleaner
// Class:           RoadSearchCloudCleaner
// 
// Description:     Calls RoadSeachCloudCleanerAlgorithm
//                  to find RoadSearchClouds.
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/07 22:00:09 $
// $Revision: 1.2 $
//

#include <memory>
#include <string>

#include "RecoTracker/RoadSearchCloudCleaner/interface/RoadSearchCloudCleaner.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "DataFormats/Common/interface/Handle.h"
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

    // Step B: create empty output collection
    std::auto_ptr<RoadSearchCloudCollection> output(new RoadSearchCloudCollection);

    // Step C: Invoke the cloud cleaning algorithm
    roadSearchCloudCleanerAlgorithm_.run(rawclouds.product(),es,*output);

    // Step D: write output to file
    e.put(output);

  }

}
