//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker
// Class:           RoadSearchTrackCandidateMaker
// 
// Description:     Calls RoadSeachTrackCandidateMakerAlgorithm
//                  to convert cleaned clouds into
//                  TrackCandidates using the 
//                  TrajectoryBuilder framework
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 15 13:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/02/22 01:16:15 $
// $Revision: 1.1 $
//

#include <memory>
#include <string>

#include "RecoTracker/RoadSearchTrackCandidateMaker/interface/RoadSearchTrackCandidateMaker.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{

  RoadSearchTrackCandidateMaker::RoadSearchTrackCandidateMaker(edm::ParameterSet const& conf) : 
    roadSearchTrackCandidateMakerAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<TrackCandidateCollection>();
  }


  // Virtual destructor needed.
  RoadSearchTrackCandidateMaker::~RoadSearchTrackCandidateMaker() { }  

  // Functions that gets called by framework every event
  void RoadSearchTrackCandidateMaker::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // Step A: Get Inputs 

    // retrieve producer name of raw CloudCollection
    std::string cleanCloudProducer = conf_.getParameter<std::string>("CleanCloudProducer");
    edm::Handle<RoadSearchCloudCollection> cleanClouds;
    e.getByLabel(cleanCloudProducer, cleanClouds);

    // Step B: create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);

    // Step C: Invoke the cloud cleaning algorithm
    roadSearchTrackCandidateMakerAlgorithm_.run(cleanClouds.product(),es,*output);

    // Step D: write output to file
    e.put(output);

  }

}
