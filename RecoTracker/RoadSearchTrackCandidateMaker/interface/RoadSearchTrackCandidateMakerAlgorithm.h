#ifndef TrackCandidateMakerAlgorithm_h
#define TrackCandidateMakerAlgorithm_h

//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker
// Class:           RoadSearchTrackCandidateMakerAlgorithm
// 
// Description:     Converts cleaned clouds into
//                  TrackCandidates using the 
//                  TrajectoryBuilder framework
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 15 13:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/02/22 01:16:14 $
// $Revision: 1.1 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

class RoadSearchTrackCandidateMakerAlgorithm 
{
 public:
  
  RoadSearchTrackCandidateMakerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchTrackCandidateMakerAlgorithm();

  /// Runs the algorithm
  void run(const RoadSearchCloudCollection* input,
	   const edm::EventSetup& es,
	   TrackCandidateCollection &output);

 private:
  edm::ParameterSet conf_;

};

#endif
