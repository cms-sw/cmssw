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
// $Date: 2006/03/29 20:10:38 $
// $Revision: 1.1 $
//

#ifndef ReadTrackCandidatesAlgorithm_h
#define ReadTrackCandidatesAlgorithm_h

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

class ReadTrackCandidatesAlgorithm 
{
 public:
  
  ReadTrackCandidatesAlgorithm(const edm::ParameterSet& conf);
  ~ReadTrackCandidatesAlgorithm();
  

  /// Runs the algorithm
  void run(const TrackCandidateCollection* input);

 private:

  edm::ParameterSet conf_;
};

#endif
