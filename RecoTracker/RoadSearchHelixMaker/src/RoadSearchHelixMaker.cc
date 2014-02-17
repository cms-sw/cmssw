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
// $Author: gutsche $
// $Date: 2007/03/07 22:04:03 $
// $Revision: 1.5 $
//

#include <memory>
#include <string>

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMaker.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/Common/interface/Handle.h"
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
    std::string trackCandidateProducer = conf_.getParameter<std::string>("TrackCandidateProducer");
    edm::Handle<TrackCandidateCollection> trackCandidates;
    e.getByLabel(trackCandidateProducer, trackCandidates);

    // Step B: create empty output collection
    std::auto_ptr<reco::TrackCollection> output(new reco::TrackCollection);

    // Step C: Invoke the cloud cleaning algorithm
    roadSearchHelixMakerAlgorithm_.run(trackCandidates.product(),es,*output);

    // Step D: write output to file
   e.put(output);

  }

}
