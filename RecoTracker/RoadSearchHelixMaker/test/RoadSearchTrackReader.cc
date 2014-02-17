//
// Package:         RecoTracker/RoadSearchHelixMaker/test
// Class:           RoadSearchTrackReader
// 
// Description:     calls RoadSearchTrackReaderAlgorithm to
//                  read and analyze TrackCandidates from the RoadSearch
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 29 20:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/06/20 09:09:19 $
// $Revision: 1.1 $
//

#include <memory>
#include <string>
#include <iostream>

#include "RecoTracker/RoadSearchHelixMaker/test/RoadSearchTrackReader.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace cms
{

  RoadSearchTrackReader::RoadSearchTrackReader(edm::ParameterSet const& conf) : 
    readTrackAlgorithm_(conf) ,
    conf_(conf)
  {
  }

  // Virtual destructor needed.
  RoadSearchTrackReader::~RoadSearchTrackReader() { }  

  // Functions that gets called by framework every event
  void RoadSearchTrackReader::analyze(const edm::Event& e, const edm::EventSetup& es)
  {
    std::string trackProducer = conf_.getParameter<std::string>("TrackProducer");

    // Get Inputs 
    edm::Handle<reco::TrackCollection> tracks;
    e.getByLabel(trackProducer, tracks);

    readTrackAlgorithm_.run(tracks.product());

  }

}
