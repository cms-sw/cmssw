//
// Package:         RecoTracker/RoadSearchHelixMaker/test
// Class:           RoadSearchTrackReaderAlgorithm
// 
// Description:     read and analyze Tracks
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 29 20:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/06/20 09:09:19 $
// $Revision: 1.1 $
//

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoTracker/RoadSearchHelixMaker/test/RoadSearchTrackReaderAlgorithm.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

RoadSearchTrackReaderAlgorithm::RoadSearchTrackReaderAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchTrackReaderAlgorithm::~RoadSearchTrackReaderAlgorithm() {
}


void RoadSearchTrackReaderAlgorithm::run(const reco::TrackCollection* input)
{
  
  edm::LogInfo("RoadSearch") << "number of Tracks: " << input->size();
  
  for ( reco::TrackCollection::const_iterator track = input->begin(); track != input->end(); ++track ) {
    edm::LogInfo("RoadSearch") << "Perigee-Parameter: transverseCurvature: " << track->transverseCurvature();
    edm::LogInfo("RoadSearch") << "Perigee-Parameter: theta: " << track->theta();
    edm::LogInfo("RoadSearch") << "Perigee-Parameter: phi0: " << track->phi0();
    edm::LogInfo("RoadSearch") << "Perigee-Parameter: d0: " << track->d0();
    edm::LogInfo("RoadSearch") << "Perigee-Parameter: dz: " << track->dz();
  }
  
}
