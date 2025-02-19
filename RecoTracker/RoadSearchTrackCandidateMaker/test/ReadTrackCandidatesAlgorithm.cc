//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker/test
// Class:           ReadTrackCandidatesAlgorithm
// 
// Description:     read and analyze TrackCandidates from the RoadSearch
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 29 20:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/11/10 21:58:22 $
// $Revision: 1.2 $
//

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoTracker/RoadSearchTrackCandidateMaker/test/ReadTrackCandidatesAlgorithm.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ReadTrackCandidatesAlgorithm::ReadTrackCandidatesAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

ReadTrackCandidatesAlgorithm::~ReadTrackCandidatesAlgorithm() {
}


void ReadTrackCandidatesAlgorithm::run(const TrackCandidateCollection* input)
{
  
   LogDebug("RoadSearch") << "number of TrackCandidates: " << input->size();

  for ( TrackCandidateCollection::const_iterator candidate = input->begin(); candidate != input->end(); ++candidate ) {
    TrackCandidate currentCandidate = *candidate;
    LogDebug("RoadSearch") << "TrajectorySeed propagation direction: " << currentCandidate.seed().direction();
    LogDebug("RoadSearch") << "PTrajectoryStateOnDet detId: " << currentCandidate.trajectoryStateOnDet().detId();
    TrackCandidate::range recHitRange = currentCandidate.recHits();
    for ( TrackCandidate::const_iterator recHit = recHitRange.first; recHit != recHitRange.second; ++recHit ) {
      LogDebug("RoadSearch") << "TrackingRecHit detId: " << recHit->geographicalId().rawId();
    }	
  }

}
