//
// Package:         RecoTracker/RoadSearchHelixMaker
// Class:           RoadSearchHelixMaker
// 
// Description:     Calls RoadSeachHelixMakerAlgorithm
//                  to find RoadSearchClouds.
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/03/22 22:43:09 $
// $Revision: 1.2 $
//

#include <memory>
#include <string>

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMaker.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{

  RoadSearchHelixMaker::RoadSearchHelixMaker(edm::ParameterSet const& conf) : 
    roadSearchHelixMakerAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<reco::TrackCollection>();
  }


  // Virtual destructor needed.
  RoadSearchHelixMaker::~RoadSearchHelixMaker() { }  

  // Functions that gets called by framework every event
  void RoadSearchHelixMaker::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // Step A: Get Inputs 

    // retrieve producer name of raw CloudCollection
    std::string cleancloudProducer = conf_.getParameter<std::string>("CleanCloudProducer");
    edm::Handle<RoadSearchCloudCollection> cleanclouds;
    e.getByLabel(cleancloudProducer, cleanclouds);

    // Step B: create empty output collection
    std::auto_ptr<reco::TrackCollection> output(new reco::TrackCollection);

    // Step C: Invoke the cloud cleaning algorithm
    roadSearchHelixMakerAlgorithm_.run(cleanclouds.product(),es,*output);

    // Step D: write output to file
   e.put(output);

  }

}
