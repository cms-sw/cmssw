//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker/test
// Class:           ReadTrackCandidates
// 
// Description:     calls ReadTrackCandidatesAlgorithm to
//                  read and analyze TrackCandidates from the RoadSearch
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 29 20:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/29 20:10:38 $
// $Revision: 1.1 $
//

#include <memory>
#include <string>
#include <iostream>

#include "RecoTracker/RoadSearchTrackCandidateMaker/test/ReadTrackCandidates.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

namespace cms
{

  ReadTrackCandidates::ReadTrackCandidates(edm::ParameterSet const& conf) : 
    readTrackCandidatesAlgorithm_(conf) ,
    conf_(conf)
  {
  }

  // Virtual destructor needed.
  ReadTrackCandidates::~ReadTrackCandidates() { }  

  // Functions that gets called by framework every event
  void ReadTrackCandidates::analyze(const edm::Event& e, const edm::EventSetup& es)
  {
    std::string trackCandidateProducer = conf_.getParameter<std::string>("TrackCandidateProducer");

    // Step A: Get Inputs 
    edm::Handle<TrackCandidateCollection> trackCandidates;
    e.getByLabel(trackCandidateProducer, trackCandidates);

    readTrackCandidatesAlgorithm_.run(trackCandidates.product());

  }

}
